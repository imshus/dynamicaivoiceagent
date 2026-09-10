"""Dynamic AI voice agent — the engine.

    browser mic ──16 kHz PCM──▶ VoiceSession ──▶ Deepgram Flux   (speech → text, turn-taking)
                                             └─▶ GPT-5.6 Luna    (the reply, streamed)
    browser ◀──24 kHz PCM──── VoiceSession ◀─── ElevenLabs       (text → speech, input-streaming)

Nothing is scripted: every reply is written live by the model from prompt.md
(or the instructions typed on the page) plus the conversation so far.

Turn-taking is driven entirely by Flux's TurnInfo events:
  StartOfTurn while the agent is talking → barge-in (reply cancelled, playback cleared)
  EagerEndOfTurn                          → the model starts writing the reply early
  TurnResumed                             → that early draft is dropped
  EndOfTurn                               → reply (reusing the early draft when the text matches)

A reply cancelled before any audio reached the browser is RETRACTED: its text
goes back into the buffer, so a caller who pauses mid-question gets one answer
to the whole question instead of two half-answers.

One voice for the whole call: a single ElevenLabs input-stream stays open from
the greeting to the end of the call, and every reply is text appended to that
same generation. A new generation per reply is what makes the voice land on a
slightly different tone, pace and loudness each time — one stream cannot.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from typing import AsyncIterator, Awaitable, Callable
from urllib.parse import quote

import websockets
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
logger = logging.getLogger("agent")
HERE = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# ── Keys ─────────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# ── Speech to text: Deepgram Flux ────────────────────────────────────────────
# flux-general-multi = conversational model with a trained end-of-turn
# detector and Hindi/English code-switching. The browser sends 16 kHz linear16.
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "flux-general-multi")
INPUT_SAMPLE_RATE = 16000
FLUX_EOT_THRESHOLD = _env_float("FLUX_EOT_THRESHOLD", 0.7)          # 0.5–0.9
FLUX_EAGER_EOT_THRESHOLD = _env_float("FLUX_EAGER_EOT_THRESHOLD", 0.5)  # 0.3–eot
FLUX_EOT_TIMEOUT_MS = _env_int("FLUX_EOT_TIMEOUT_MS", 5000)
FLUX_LANGUAGES = {"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"}
FLUX_LANGUAGE_HINTS = [h.strip().lower() for h in
                       os.getenv("FLUX_LANGUAGE_HINTS", "hi,en").split(",") if h.strip()]
# Keyterm prompting: the words this helpline turns on, biased so Flux stops
# hearing "cash tone" for "colorstone" or "cast" for "karat". The model can only
# understand a caller whose words arrived intact, so this is the first place a
# misunderstood question is fixed — before any prompt wording. Sent as REPEATED
# keyterm= parameters (one comma-joined value would be taken as a single literal
# term and boost nothing). Deepgram's limit is 500 tokens across all terms.
DEEPGRAM_KEYTERMS = [t.strip() for t in os.getenv(
    "DEEPGRAM_KEYTERMS",
    "MRPscan,MRP,bullion,MCX,RTGS,cash rate,karat,purity,tunch,gold rate,"
    "making charges,labour charge,wastage,gross weight,net weight,"
    "diamond,packet code,sieve,clarity,colorstone,stone rate,"
    "item code,masters,dashboard settings,scanner,tag,"
    "e-invoice,GST,wishlist,employee manager,set permission,"
    "dashboard matrices,active account,password manager"
).split(",") if t.strip()]

# ── The brain: GPT-5.6 Luna ──────────────────────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None
# "none" = no thinking tokens, so the first word arrives phone-fast.
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "none")
# GPT-5.x rejects temperature/max_tokens; set this only for a classic model.
OPENAI_CLASSIC_SAMPLING = _env_bool("OPENAI_CLASSIC_SAMPLING", False)
LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 300)
MAX_HISTORY_TURNS = _env_int("MAX_HISTORY_TURNS", 20)
# Start the model on Flux's EagerEndOfTurn so its latency is hidden behind the
# last ~200 ms of the caller's turn. One extra LLM call per false eager.
SPECULATIVE_LLM = _env_bool("SPECULATIVE_LLM", True)
# Waiting for a step to be done is not the same as waiting for an answer. When
# the agent has asked whether the caller is there and nothing comes back, it
# checks in rather than sitting mute until the line times out. Twice, then it
# leaves them alone — a caller who is genuinely busy must not be nagged.
# 0 turns it off.
NUDGE_AFTER_SECONDS = _env_float("NUDGE_AFTER_SECONDS", 12.0)
NUDGE_MAX = _env_int("NUDGE_MAX", 2)
NUDGE_INSTRUCTION = os.getenv(
    "NUDGE_INSTRUCTION",
    "The caller has said nothing for a while. They are probably still doing the last step, or "
    "did not hear you. Check in once, in one short sentence, in the language you have been "
    "speaking: ask whether they have done it, and offer to say it again if they are stuck. Do "
    "not repeat the whole explanation and do not move on to the next step.")
LLM_ERROR_REPLY = os.getenv("LLM_ERROR_REPLY", "Sorry, I missed that. Could you say it again?")
_PROMPT_RAW = os.getenv("PROMPT_FILE", "prompt.md")
PROMPT_FILE = _PROMPT_RAW if os.path.isabs(_PROMPT_RAW) else os.path.join(HERE, _PROMPT_RAW)
DEFAULT_PROMPT = ("You are a friendly voice assistant on a live call. Reply the way the "
                  "caller speaks, in one to three short spoken sentences, with no lists, "
                  "markdown or emojis.")
# What the agent knows — the MRPscan FAQ, in English and Hindi. It is appended
# to the system prompt on every call (even when the client sends its own
# instructions), so the answers come from approved text rather than invention.
# The file is read fresh per call: edit it and the next caller gets the change.
# Point KNOWLEDGE_FILE at another file, or leave it empty, for an agent that
# should not know any of this.
_KNOWLEDGE_RAW = os.getenv("KNOWLEDGE_FILE", "faq.md")
KNOWLEDGE_FILE = ("" if not _KNOWLEDGE_RAW.strip() else
                  _KNOWLEDGE_RAW if os.path.isabs(_KNOWLEDGE_RAW)
                  else os.path.join(HERE, _KNOWLEDGE_RAW))

# ── Agent gender: the voice and the model's own grammar must agree ───────────
# Hindi verbs carry the speaker's gender ("बोल रही हूँ" / "बोल रहा हूँ"), so a
# male voice reading feminine text sounds wrong. The page picks female or male
# per call; that choice selects the voice, the name, the greeting, and one line
# appended to the system prompt.
VOICE_IDS = {
    "female": (os.getenv("ELEVENLABS_VOICE_ID_FEMALE") or os.getenv("ELEVENLABS_VOICE_ID")
               or "EXAVITQu4vr4xnSDxMaL"),
    "male": os.getenv("ELEVENLABS_VOICE_ID_MALE") or "onwK4e9ZLuTAKqWW03F9",
}
AGENT_NAMES = {"female": os.getenv("AGENT_NAME_FEMALE", "Priya").strip(),
               "male": os.getenv("AGENT_NAME_MALE", "Arjun").strip()}
DEFAULT_GENDER = os.getenv("AGENT_GENDER", "female").strip().lower()
if DEFAULT_GENDER not in VOICE_IDS:
    DEFAULT_GENDER = "female"

# ── Text to speech: ElevenLabs input-streaming ───────────────────────────────
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
OUTPUT_SAMPLE_RATE = _env_int("OUTPUT_SAMPLE_RATE", 24000)   # pcm_16000 | pcm_22050 | pcm_24000
# Constant delivery: stability 1.0 = one steady pitch and loudness; style 0 and
# speaker boost off — both make the level drift; speed fixed for the whole call.
ELEVENLABS_STABILITY = _env_float("ELEVENLABS_STABILITY", 1.0)
ELEVENLABS_SIMILARITY = _env_float("ELEVENLABS_SIMILARITY", 0.85)
ELEVENLABS_STYLE = _env_float("ELEVENLABS_STYLE", 0.0)
ELEVENLABS_SPEAKER_BOOST = _env_bool("ELEVENLABS_SPEAKER_BOOST", False)
TTS_SPEED = _env_float("TTS_SPEED", 1.0)
# Generation pacing inside the one stream. Text accumulates until the schedule
# threshold is reached OR we flush. Flushing at every sentence end keeps the
# first audio of each reply fast; set false to let only the schedule (and the
# end of the reply) trigger generation — larger pieces, a little more wait.
TTS_CHUNK_SCHEDULE = [50, 80, 120, 150]
TTS_FLUSH_EVERY_SENTENCE = _env_bool("TTS_FLUSH_EVERY_SENTENCE", True)
TTS_INACTIVITY_TIMEOUT = 180        # the most ElevenLabs allows
TTS_KEEPALIVE_SECONDS = 15          # a space now and then keeps the stream (and the voice) alive
TTS_IDLE_RESYNC_SECONDS = 1.5       # after this much silence from ElevenLabs, everything sent counts as voiced

# ── Turn-taking ──────────────────────────────────────────────────────────────
BARGE_IN_ENABLED = _env_bool("BARGE_IN_ENABLED", True)
BARGE_IN_MIN_CHARS = _env_int("BARGE_IN_MIN_CHARS", 6)
# Share of the caller's words that also occur in what the agent just said,
# above which the "speech" is treated as the agent's own echo.
BARGE_IN_ECHO_OVERLAP = _env_float("BARGE_IN_ECHO_OVERLAP", 0.6)
# A phone on speaker has no echo canceller on that path: the microphone hears
# the agent and Flux transcribes it as if the caller had spoken. Besides the
# overlap share, a transcript arriving while the agent talks must carry at
# least this many words the agent did NOT say before it counts as the caller
# cutting in — a fragment of the agent's own sentence carries none.
BARGE_IN_MIN_NEW_WORDS = _env_int("BARGE_IN_MIN_NEW_WORDS", 2)
# GREETING may use {name}; GREETING_FEMALE / GREETING_MALE override it per
# gender (a Hindi greeting needs that, since its verbs are gendered too).
GREETING = os.getenv("GREETING", "").strip()
GREETINGS = {g: (os.getenv(f"GREETING_{g.upper()}") or GREETING).strip() for g in VOICE_IDS}


def gender_line(gender: str) -> str:
    """The one line that tells the model who it is speaking as."""
    who, forms = ("a woman", "feminine") if gender == "female" else ("a man", "masculine")
    return (f"Your name is {AGENT_NAMES[gender]}. You are {who}: in Hindi and Hinglish use "
            f"{forms} forms when speaking about yourself, and address the caller with the "
            f"forms that match how they speak.")


def _read_file(path: str) -> str:
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except OSError as e:
        logger.warning(f"Could not read {path}: {e!r}")
        return ""


def load_prompt() -> str:
    """Read prompt.md fresh (so an edit applies to the next call, no restart)."""
    return _read_file(PROMPT_FILE) or DEFAULT_PROMPT


def load_knowledge() -> str:
    """Read the FAQ fresh, for the same reason. Empty when there is no file."""
    return _read_file(KNOWLEDGE_FILE)


def build_system_prompt(instructions: str | None, gender: str) -> str:
    """Persona (the client's own, or prompt.md), then who the agent is, then
    everything it knows."""
    parts = [(instructions or "").strip() or load_prompt(), gender_line(gender)]
    knowledge = load_knowledge()
    if knowledge:
        parts.append(knowledge)
    return "\n\n".join(parts)


def deepgram_url() -> str:
    eot = min(max(FLUX_EOT_THRESHOLD, 0.5), 0.9)
    eager = min(max(FLUX_EAGER_EOT_THRESHOLD, 0.3), 0.9, eot)
    timeout = min(max(FLUX_EOT_TIMEOUT_MS, 500), 60000)
    url = (f"wss://api.deepgram.com/v2/listen?model={DEEPGRAM_MODEL}"
           f"&encoding=linear16&sample_rate={INPUT_SAMPLE_RATE}"
           f"&eot_threshold={eot}&eager_eot_threshold={eager}&eot_timeout_ms={timeout}")
    if DEEPGRAM_MODEL == "flux-general-multi":
        url += "".join(f"&language_hint={h}" for h in FLUX_LANGUAGE_HINTS if h in FLUX_LANGUAGES)
    url += "".join(f"&keyterm={quote(t)}" for t in DEEPGRAM_KEYTERMS)
    return url


def tts_url(voice_id: str) -> str:
    return (f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
            f"?model_id={ELEVENLABS_MODEL}&output_format=pcm_{OUTPUT_SAMPLE_RATE}"
            f"&inactivity_timeout={TTS_INACTIVITY_TIMEOUT}")


async def open_tts_ws(voice_id: str):
    """Connect to ElevenLabs and send the config frame, ready to take text."""
    ws = await websockets.connect(tts_url(voice_id), additional_headers={"xi-api-key": ELEVENLABS_API_KEY})
    await ws.send(json.dumps({
        "text": " ",
        "voice_settings": {
            "stability": ELEVENLABS_STABILITY,
            "similarity_boost": ELEVENLABS_SIMILARITY,
            "style": ELEVENLABS_STYLE,
            "use_speaker_boost": ELEVENLABS_SPEAKER_BOOST,
            "speed": TTS_SPEED,
        },
        "generation_config": {"chunk_length_schedule": TTS_CHUNK_SCHEDULE},
    }))
    return ws


_llm_client: AsyncOpenAI | None = None


def get_llm() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(api_key=OPENAI_API_KEY or "missing", base_url=OPENAI_BASE_URL)
    return _llm_client


def llm_params() -> dict:
    if OPENAI_CLASSIC_SAMPLING:
        return {"max_tokens": LLM_MAX_TOKENS, "temperature": 0.7}
    return {"max_completion_tokens": LLM_MAX_TOKENS, "reasoning_effort": OPENAI_REASONING_EFFORT}


# ── Splitting the token stream into speakable pieces ─────────────────────────
# A sentence ends at . ! ? or the Devanagari danda, followed by whitespace (the
# whitespace requirement keeps "3.5" and "Rs.500" whole). For the FIRST piece
# of a reply a clause boundary is enough, which is what cuts time-to-audio.
_SENT_RE = re.compile(r'^(.*?[.!?।]["\')\]]*)\s+', re.S)
_CLAUSE_RE = re.compile(r'^(.*?[,;:—])\s+', re.S)
_MARKUP_RE = re.compile(r'[*#`]+')
_ARROW_RE = re.compile(r'\s*(?:→|->|➜|»)\s*')


def next_chunk(buf: str, allow_clause: bool, clause_min_chars: int = 20) -> tuple[str, str]:
    """Return (chunk ready to speak, remaining buffer); chunk is '' if none yet."""
    nl = buf.find("\n")
    m = _SENT_RE.match(buf)
    if m and (nl < 0 or m.end() <= nl + 1):       # sentence ends before the newline
        return m.group(1).strip(), buf[m.end():]
    if nl >= 0:                                    # a line break is a boundary too
        return buf[:nl].strip(), buf[nl + 1:]
    if allow_clause and len(buf) >= clause_min_chars:
        m = _CLAUSE_RE.match(buf)
        if m:
            return m.group(1).strip(), buf[m.end():]
    return "", buf


def clean_for_speech(text: str) -> str:
    """Strip markup, and turn exclamation marks into full stops: "Hello!" is
    what makes the voice jump bright and loud for a sentence and then settle.
    Menu arrows copied out of the FAQ become commas, so a step never reaches
    the voice as a symbol."""
    text = _MARKUP_RE.sub("", text)
    text = _ARROW_RE.sub(", ", text)
    text = re.sub(r"([.?])!+", r"\1", text)
    text = re.sub(r"!+", ".", text)
    text = re.sub(r"\s+([,.?])", r"\1", text)
    return text.strip()


def _nonspace(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


async def _static_tokens(text: str) -> AsyncIterator[str]:
    yield text


class _Speculation:
    """An LLM reply started on EagerEndOfTurn, buffered until EndOfTurn confirms it."""

    def __init__(self, text: str, task: asyncio.Task, queue: asyncio.Queue):
        self.text, self.task, self.queue = text, task, queue

    def cancel(self):
        if not self.task.done():
            self.task.cancel()

    async def tokens(self) -> AsyncIterator[str]:
        while True:
            tok = await self.queue.get()
            if tok is None:
                return
            yield tok


class _Reply:
    """State of one reply in flight (greeting, or the answer to one caller turn)."""

    def __init__(self, user_text: str | None, source: AsyncIterator[str],
                 extra_task: asyncio.Task | None, turn_end: float | None):
        self.user_text = user_text
        self.source = source
        self.extra_task = extra_task
        self.turn_end = turn_end
        self.full_text = ""
        self.spoken: list[str] = []
        self.audio_started = False
        self.first_token_at: float | None = None
        self.retracted = False
        self.user_appended = False
        self.ns_start = 0            # stream position (non-space chars) where this reply's text begins
        self.hist_mark = 0           # len(history) before this reply touched it


class VoiceSession:
    """One live conversation. `send_audio(bytes)` and `send_json(dict)` deliver
    to the browser; feed_audio() takes the caller's 16 kHz PCM frames."""

    def __init__(self, send_audio: Callable[[bytes], Awaitable[None]],
                 send_json: Callable[[dict], Awaitable[None]],
                 instructions: str | None = None, gender: str | None = None):
        self._send_audio = send_audio
        self._send_json = send_json
        g = (gender or "").strip().lower()
        self.gender = g if g in VOICE_IDS else DEFAULT_GENDER
        self.voice_id = VOICE_IDS[self.gender]
        self.name = AGENT_NAMES[self.gender]
        self.system_prompt = build_system_prompt(instructions, self.gender)
        self.history: list[dict] = []
        self.active = True
        self.turn_buffer = ""
        self.reply_task: asyncio.Task | None = None
        self._reply_state: _Reply | None = None
        self._spec: _Speculation | None = None
        self._deferred: asyncio.Task | None = None   # a confirmation heard mid-reply
        self._nudge_task: asyncio.Task | None = None
        self._quiet_since: float | None = None       # when the line last went quiet
        self._nudges_sent = 0
        self._turn_end: float | None = None
        self.playback_until = 0.0
        self._agent_text = ""            # what the agent is saying now (echo check)
        self._spoken_before = ""         # the reply before it, still ringing in the room
        self._dg_ws = None
        self._dg_task: asyncio.Task | None = None
        # The one ElevenLabs stream for the whole call, and where we are in it.
        self._tts_ws = None
        self._tts_rx: asyncio.Task | None = None
        self._tts_keepalive: asyncio.Task | None = None
        self._tts_lock = asyncio.Lock()
        self._sent_ns = 0                # non-space chars sent to the stream since it opened
        self._voiced_ns = 0              # non-space chars whose audio has come back
        self._discard_until_ns = 0       # audio for chars below this position is dropped (barge-in)
        self._tts_last_send = 0.0
        self._tts_last_audio = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self):
        logger.info("Session start: %s voice %s, name %s", self.gender, self.voice_id, self.name)
        self._dg_task = asyncio.create_task(self._deepgram_loop())
        try:
            await self._tts_open()
        except Exception as e:
            logger.error(f"ElevenLabs stream could not be opened: {e!r} — will retry on the first reply")
        self._tts_keepalive = asyncio.create_task(self._tts_keepalive_loop())
        self._nudge_task = asyncio.create_task(self._nudge_loop())
        await self._send_json({"type": "ready", "sample_rate": OUTPUT_SAMPLE_RATE,
                               "gender": self.gender, "name": self.name})
        greeting = GREETINGS[self.gender].replace("{name}", self.name)
        if greeting:
            self._launch_reply(None, _static_tokens(greeting))

    async def feed_audio(self, pcm: bytes):
        ws = self._dg_ws
        if ws is None:
            return
        try:
            await ws.send(pcm)
        except Exception as e:
            logger.warning(f"Could not forward audio to Deepgram: {e!r}")

    async def interrupt(self, reason: str = "caller spoke"):
        """
        The client heard its own user start talking and says so.

        A phone hears the agent through its own speaker, so waiting for Flux
        to call it a turn is both slow and unreliable — the client knows
        first. This stops the reply exactly as Flux's StartOfTurn does; what
        the caller is saying then arrives as audio in the usual way.
        """
        if not self._agent_busy():
            return
        await self._barge_in(f"[client: {reason}]")

    async def close(self):
        self.active = False
        self._drop_speculation("session closed")
        self._cancel_deferred()
        for task in (self.reply_task, self._dg_task, self._tts_keepalive, self._nudge_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait({task}, timeout=2)
                except Exception:
                    pass
        ws, self._tts_ws = self._tts_ws, None
        if ws is not None:
            try:
                await ws.send(json.dumps({"text": ""}))      # end of stream
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
        if self._tts_rx is not None and not self._tts_rx.done():
            self._tts_rx.cancel()
            try:
                await asyncio.wait({self._tts_rx}, timeout=2)
            except Exception:
                pass
        logger.info("Session closed (%d messages)", len(self.history))

    # ── Deepgram Flux ────────────────────────────────────────────────────────
    async def _deepgram_loop(self):
        url = deepgram_url()
        attempt = 0
        while self.active:
            ws = None
            try:
                ws = await websockets.connect(
                    url, additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"})
                self._dg_ws = ws
                attempt = 0
                logger.info("Deepgram Flux connected (%s)", DEEPGRAM_MODEL)
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                    except Exception:
                        continue
                    await self._on_deepgram(data)
            except asyncio.CancelledError:
                raise
            except websockets.exceptions.InvalidStatus as e:
                status = getattr(getattr(e, "response", None), "status_code", "?")
                logger.error(f"Deepgram refused the connection (HTTP {status}) — check DEEPGRAM_API_KEY")
            except websockets.exceptions.ConnectionClosed as e:
                logger.info(f"Deepgram socket closed: {e}")
            except Exception as e:
                logger.error(f"Deepgram listener error: {e!r}")
            finally:
                self._dg_ws = None
                if ws is not None:
                    try:
                        await ws.close()
                    except Exception:
                        pass
            if self.active:
                attempt += 1
                delay = min(0.5 * attempt, 3.0)
                logger.info(f"Reconnecting to Deepgram in {delay:.1f}s")
                await asyncio.sleep(delay)

    async def _on_deepgram(self, data: dict):
        t = data.get("type")
        if t == "TurnInfo":
            await self._on_turn(data)
        elif t in ("Error", "FatalError"):
            logger.error(f"Deepgram rejected the stream — {data.get('code', '')}: "
                         f"{data.get('description') or data.get('message') or data}")
        elif t == "Connected":
            logger.debug("Deepgram request %s", data.get("request_id", ""))

    async def _on_turn(self, data: dict):
        event = data.get("event")
        transcript = (data.get("transcript") or "").strip()
        busy = self._agent_busy()

        if event in ("StartOfTurn", "Update"):
            if transcript and busy and BARGE_IN_ENABLED:
                if self._looks_like_echo(transcript):
                    logger.info(f"[{event}] {transcript!r} → own echo, ignored")
                elif len(transcript) >= BARGE_IN_MIN_CHARS:
                    await self._barge_in(transcript)
            return

        if event == "EagerEndOfTurn":
            if transcript and SPECULATIVE_LLM and not (busy and self._looks_like_echo(transcript)):
                self._speculate(transcript)
            return

        if event == "TurnResumed":
            self._drop_speculation("TurnResumed")
            return

        if event == "EndOfTurn":
            if not transcript:
                return
            if busy and self._sounds_like_own_words(transcript):
                logger.info(f"[EndOfTurn] {transcript!r} → own echo, dropped")
                return
            if busy:
                # Words of the caller's own, said while the agent was still
                # talking — in a walkthrough this is "haan, ho gaya" arriving
                # on top of the step. Too small to interrupt for, but it is a
                # real turn: hold it and answer the moment the line is quiet,
                # instead of making the caller say it twice.
                self.turn_buffer = (self.turn_buffer + " " + transcript).strip()
                self._caller_spoke()
                logger.info(f"[EndOfTurn while speaking] {transcript!r} → answered after this reply")
                if self._deferred is None or self._deferred.done():
                    self._deferred = asyncio.create_task(self._turn_after_playback())
                return
            self._turn_end = time.monotonic()
            self._caller_spoke()
            text = (self.turn_buffer + " " + transcript).strip()
            self.turn_buffer = ""
            logger.info(f"[EndOfTurn] {text!r}")
            spec, self._spec = self._spec, None
            if spec is not None and spec.text == text:
                logger.info("[latency] early draft reused")
                self._launch_reply(text, spec.tokens(), extra_task=spec.task)
            else:
                if spec is not None:
                    spec.cancel()
                    logger.info("early draft discarded (text changed)")
                self._launch_reply(text, self._llm_stream(self._messages(text)))

    async def _nudge_loop(self):
        """The caller went quiet after being asked something. Check in, at most
        NUDGE_MAX times, then leave the line alone."""
        try:
            while self.active:
                await asyncio.sleep(0.25)
                if NUDGE_AFTER_SECONDS <= 0 or not self.history:
                    continue
                if self._agent_busy() or self.turn_buffer.strip() or self._spec is not None:
                    self._quiet_since = None      # someone is mid-turn
                    continue
                if self._quiet_since is None:
                    self._quiet_since = time.monotonic()
                    continue
                if (self._nudges_sent >= NUDGE_MAX
                        or time.monotonic() - self._quiet_since < NUDGE_AFTER_SECONDS):
                    continue
                self._nudges_sent += 1
                self._quiet_since = None
                self._turn_end = None             # nothing to measure latency against
                logger.info(f"[nudge {self._nudges_sent}/{NUDGE_MAX}] caller quiet for "
                            f"{NUDGE_AFTER_SECONDS:.0f}s — checking in")
                self._launch_reply(None, self._llm_stream(self._nudge_messages()))
        except asyncio.CancelledError:
            pass

    def _caller_spoke(self):
        """A real turn from the caller: they are with us, so start counting again."""
        self._nudges_sent = 0
        self._quiet_since = None

    async def _turn_after_playback(self):
        """Answer a held confirmation as soon as the agent stops speaking."""
        try:
            while self._agent_busy():
                await asyncio.sleep(0.05)
            text = self.turn_buffer.strip()
            if not text:
                return
            self.turn_buffer = ""
            self._turn_end = time.monotonic()
            logger.info(f"[held turn] {text!r}")
            self._launch_reply(text, self._llm_stream(self._messages(text)))
        except asyncio.CancelledError:
            pass

    def _cancel_deferred(self):
        """A fresher turn won: whatever was held is part of it now, or gone."""
        task, self._deferred = self._deferred, None
        if task is not None and not task.done():
            task.cancel()

    # ── turn-taking helpers ──────────────────────────────────────────────────
    def _audio_owed(self) -> int:
        """Non-space chars sent to ElevenLabs whose audio is still to come and will be played."""
        return self._sent_ns - max(self._voiced_ns, self._discard_until_ns)

    def _agent_busy(self) -> bool:
        task = self.reply_task
        return ((task is not None and not task.done())
                or time.monotonic() < self.playback_until
                or self._audio_owed() > 0)

    def _looks_like_echo(self, transcript: str) -> bool:
        """
        Is this the agent hearing itself?

        The room can still be carrying the previous reply as well as the one
        being spoken, so both count as "what the agent said". A transcript is
        the agent's own echo when most of its words are the agent's, or when
        it brings too few words of its own to be the caller cutting in.
        """
        if self._sounds_like_own_words(transcript):
            return True
        words = re.findall(r"\w+", transcript.lower())
        said_words = set(re.findall(r"\w+", (self._agent_text + " " + self._spoken_before).lower()))
        known = sum(w in said_words for w in words)
        return (len(words) - known) < BARGE_IN_MIN_NEW_WORDS

    def _sounds_like_own_words(self, transcript: str) -> bool:
        """The narrower half of the test: are most of these the agent's own
        words coming back? A caller's "haan" shares nothing with the step just
        read out, so it fails here — which is what lets a one-word confirmation
        be held as a turn while still never interrupting the agent."""
        said = (self._agent_text + " " + self._spoken_before).strip()
        if not said:
            return False
        words = re.findall(r"\w+", transcript.lower())
        if not words:
            return False
        said_words = set(re.findall(r"\w+", said.lower()))
        return sum(w in said_words for w in words) / len(words) >= BARGE_IN_ECHO_OVERLAP

    async def _barge_in(self, heard: str):
        st, task = self._reply_state, self.reply_task
        logger.info(f"[barge-in] {heard!r}")
        # The caller is properly talking now; anything held for later belongs
        # to this turn, which arrives as its own EndOfTurn.
        self._cancel_deferred()
        running = task is not None and not task.done()
        if running:
            task.cancel()
        if st is not None and not st.audio_started and st.user_text and not st.retracted:
            # Nothing was heard yet: this is the caller finishing their thought.
            st.retracted = True
            self.turn_buffer = (st.user_text + " " + self.turn_buffer).strip()
            if not running:
                # The text was already written and sent, but none of it was
                # heard — forget it entirely, as if the question had not been asked yet.
                del self.history[st.hist_mark:]
                self._reply_state = None
                await self._send_json({"type": "retract"})
            logger.info("reply retracted — text kept for the merged turn")
        elif st is not None and st.audio_started and not running:
            # The text was complete before the caller cut in: keep only what they heard.
            await self._trim_to_heard(st)
        # Everything sent so far is dead: drop its audio as it comes back, and
        # push it through ElevenLabs now so the next reply is not queued behind it.
        async with self._tts_lock:
            owed = self._sent_ns - self._voiced_ns
            self._discard_until_ns = self._sent_ns
        if owed > 0 and self._tts_ws is not None:
            try:
                await self._tts_send(" ", flush=True)
            except Exception as e:
                logger.warning(f"flush after barge-in failed: {e!r}")
        self.playback_until = 0.0
        await self._send_json({"type": "clear"})

    def _speculate(self, transcript: str):
        text = (self.turn_buffer + " " + transcript).strip()
        if self._spec is not None:
            if self._spec.text == text:
                return
            self._spec.cancel()
        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(self._llm_to_queue(self._messages(text), queue))
        self._spec = _Speculation(text, task, queue)
        logger.info(f"[EagerEndOfTurn] drafting early: {text!r}")

    def _drop_speculation(self, why: str):
        if self._spec is not None:
            logger.info(f"early draft dropped ({why})")
            self._spec.cancel()
            self._spec = None

    async def _llm_to_queue(self, messages: list[dict], queue: asyncio.Queue):
        try:
            async for tok in self._llm_stream(messages):
                queue.put_nowait(tok)
        finally:
            queue.put_nowait(None)

    # ── the brain ────────────────────────────────────────────────────────────
    def _nudge_messages(self) -> list[dict]:
        """The conversation so far, plus a note that the line has gone quiet.
        The note is never stored, so it cannot pile up over a long call."""
        return ([{"role": "system", "content": self.system_prompt}]
                + self.history[-2 * MAX_HISTORY_TURNS:]
                + [{"role": "system", "content": NUDGE_INSTRUCTION}])

    def _messages(self, user_text: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        msgs += self.history[-2 * MAX_HISTORY_TURNS:]
        msgs.append({"role": "user", "content": user_text})
        return msgs

    async def _llm_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        try:
            stream = await get_llm().chat.completions.create(
                model=OPENAI_MODEL, messages=messages, stream=True, **llm_params())
        except Exception as e:
            logger.error(f"LLM request failed: {e!r}")
            yield LLM_ERROR_REPLY
            return
        try:
            async for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                if delta is not None and delta.content:
                    yield delta.content
        except Exception as e:
            logger.error(f"LLM stream broke: {e!r}")
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    pass

    # ── the reply pipeline: tokens → the ElevenLabs stream ───────────────────
    def _launch_reply(self, user_text: str | None, source: AsyncIterator[str],
                      extra_task: asyncio.Task | None = None):
        self._cancel_deferred()
        prev = self.reply_task
        st = _Reply(user_text, source, extra_task, self._turn_end if user_text else None)
        self._reply_state = st
        self.reply_task = asyncio.create_task(self._reply_after(prev, st))

    async def _reply_after(self, prev: asyncio.Task | None, st: _Reply):
        # _reply_state deliberately outlives this task: the text is finished
        # before its audio has all come back, and a barge-in in that window
        # still needs to know whether anything of this reply was heard.
        if prev is not None and not prev.done():
            prev.cancel()
            try:
                await asyncio.wait({prev}, timeout=2)
            except Exception:
                pass
        await self._run_reply(st)

    async def _run_reply(self, st: _Reply):
        st.hist_mark = len(self.history)
        if st.user_text is not None:
            self.history.append({"role": "user", "content": st.user_text})
            st.user_appended = True
            await self._send_json({"type": "user", "text": st.user_text})
        self._agent_text = ""
        st.ns_start = self._sent_ns
        try:
            await self._feed_tts(st)
            text = clean_for_speech(st.full_text)
            if text:
                self.history.append({"role": "assistant", "content": text})
            await self._send_json({"type": "agent", "text": text, "final": True})
            self._log_latency(st, "reply text complete")
        except asyncio.CancelledError:
            await self._finish_cancelled(st)
            raise
        except Exception as e:
            logger.error(f"Reply failed: {e!r}")
            heard = " ".join(st.spoken).strip()
            if heard:
                self.history.append({"role": "assistant", "content": heard})
            await self._send_json({"type": "error", "text": f"Reply failed: {e}"})
        finally:
            if st.extra_task is not None and not st.extra_task.done():
                st.extra_task.cancel()

    async def _finish_cancelled(self, st: _Reply):
        if st.retracted:
            del self.history[st.hist_mark:]
            if self._reply_state is st:
                self._reply_state = None
            await self._send_json({"type": "retract"})
            return
        heard = self._heard_text(st)
        if heard:
            self.history.append({"role": "assistant", "content": heard})
        if self._reply_state is st:
            self._reply_state = None
        await self._send_json({"type": "agent", "text": heard, "final": True, "interrupted": True})

    def _heard_text(self, st: _Reply) -> str:
        """The sentences of this reply that ElevenLabs had voiced when it was cut
        off — the text was written well ahead of the audio, so this is what the
        caller can actually have heard."""
        voiced = self._voiced_ns - st.ns_start
        out, acc = [], 0
        for chunk in st.spoken:
            if acc >= voiced:
                break
            out.append(chunk)
            acc += _nonspace(chunk)
        return " ".join(out).strip()

    async def _trim_to_heard(self, st: _Reply):
        idx = st.hist_mark + (1 if st.user_appended else 0)
        heard = self._heard_text(st)
        if self._reply_state is st:
            self._reply_state = None
        if idx >= len(self.history) or self.history[idx].get("role") != "assistant" \
                or self.history[idx].get("content") == heard:
            return
        if heard:
            self.history[idx]["content"] = heard
        else:
            del self.history[idx]
        await self._send_json({"type": "agent", "text": heard, "final": True, "interrupted": True})

    async def _feed_tts(self, st: _Reply):
        buf = ""
        try:
            async for tok in st.source:
                if not tok:
                    continue
                if st.first_token_at is None:
                    st.first_token_at = time.monotonic()
                    self._log_latency(st, "first token")
                buf += tok
                st.full_text += tok
                while True:
                    # Whole sentences only: a clause fragment ("Hello there,")
                    # voiced on its own is a seam where the tone can shift.
                    chunk, rest = next_chunk(buf, allow_clause=False)
                    if not chunk and rest == buf:
                        break
                    buf = rest
                    if chunk:
                        await self._say(st, chunk)
            if buf.strip():
                await self._say(st, buf)
            if not TTS_FLUSH_EVERY_SENTENCE:
                await self._tts_send(" ", flush=True)     # voice whatever is still buffered
        finally:
            try:
                await st.source.aclose()
            except Exception:
                pass

    async def _say(self, st: _Reply, chunk: str):
        chunk = clean_for_speech(chunk)
        if not chunk:
            return
        await self._tts_send(chunk + " ", flush=TTS_FLUSH_EVERY_SENTENCE)
        st.spoken.append(chunk)
        self._agent_text = " ".join(st.spoken)
        await self._send_json({"type": "agent", "text": self._agent_text, "final": False})

    async def _play(self, pcm: bytes):
        now = time.monotonic()
        self.playback_until = max(self.playback_until, now + 0.05) + len(pcm) / 2 / OUTPUT_SAMPLE_RATE
        await self._send_audio(pcm)

    def _log_latency(self, st: _Reply, label: str):
        if st.turn_end is not None:
            logger.info(f"[latency] {label}: {(time.monotonic() - st.turn_end) * 1000:.0f} ms after end of turn")

    # ── the one ElevenLabs stream ────────────────────────────────────────────
    async def _tts_open(self):
        ws = await open_tts_ws(self.voice_id)
        self._tts_ws = ws
        self._sent_ns = self._voiced_ns = self._discard_until_ns = 0
        self._tts_last_send = self._tts_last_audio = time.monotonic()
        self._tts_rx = asyncio.create_task(self._tts_receive(ws))
        logger.info("ElevenLabs stream open (voice %s) — stays open for the whole call", self.voice_id)

    async def _tts_send(self, text: str, flush: bool = False):
        async with self._tts_lock:
            ws = self._tts_ws
            if ws is None or ws.close_code is not None:
                logger.info("ElevenLabs stream is closed — reopening")
                await self._tts_open()
                ws = self._tts_ws
            msg = {"text": text}
            if flush:
                msg["flush"] = True
            self._sent_ns += _nonspace(text)
            self._tts_last_send = time.monotonic()
            await ws.send(json.dumps(msg))

    async def _tts_receive(self, ws):
        try:
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                audio = data.get("audio")
                if audio:
                    await self._on_tts_audio(base64.b64decode(audio),
                                             data.get("alignment") or data.get("normalizedAlignment"))
                elif data.get("error") or data.get("message"):
                    logger.error(f"ElevenLabs: {data}")
                if data.get("isFinal"):
                    break
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"ElevenLabs stream closed: {e}")
        except Exception as e:
            logger.error(f"ElevenLabs receiver error: {e!r}")
        finally:
            if self._tts_ws is ws:
                self._tts_ws = None

    async def _on_tts_audio(self, pcm: bytes, alignment: dict | None):
        """Route one audio piece: drop what belongs to a cancelled reply (by
        character position, cutting inside the piece when needed), then play."""
        if not pcm:
            return
        self._tts_last_audio = time.monotonic()
        st = self._reply_state
        chars = (alignment or {}).get("chars") or []
        if chars:
            starts = (alignment or {}).get("charStartTimesMs") or []
            begin = self._voiced_ns
            n = _nonspace("".join(chars))
            self._voiced_ns = begin + n
            if self._discard_until_ns > begin:
                drop = min(n, self._discard_until_ns - begin)
                if drop >= n:
                    return
                seen, idx = 0, 0
                for i, c in enumerate(chars):
                    if not c.isspace():
                        if seen == drop:
                            idx = i
                            break
                        seen += 1
                if starts and idx < len(starts):
                    cut_ms = max(0, starts[idx] - starts[0])
                    pcm = pcm[int(cut_ms / 1000 * OUTPUT_SAMPLE_RATE) * 2:]
                if not pcm:
                    return
            if st is not None and not st.audio_started and self._voiced_ns > st.ns_start:
                st.audio_started = True
                self._log_latency(st, "first audio")
        else:
            if self._discard_until_ns > self._voiced_ns:
                return                       # nothing to go by; still inside cancelled text
            if st is not None and st.spoken and not st.audio_started:
                st.audio_started = True
                self._log_latency(st, "first audio")
        await self._play(pcm)

    async def _tts_keepalive_loop(self):
        """Keep the stream open across silences, and re-sync the position
        counters once ElevenLabs has gone quiet after a reply."""
        try:
            while self.active:
                await asyncio.sleep(1.0)
                now = time.monotonic()
                if self._tts_ws is None:
                    continue
                if (self._sent_ns > self._voiced_ns
                        and now - self._tts_last_audio > TTS_IDLE_RESYNC_SECONDS
                        and now - self._tts_last_send > TTS_IDLE_RESYNC_SECONDS):
                    self._voiced_ns = self._sent_ns
                    self._discard_until_ns = min(self._discard_until_ns, self._sent_ns)
                if now - self._tts_last_send >= TTS_KEEPALIVE_SECONDS:
                    try:
                        await self._tts_send(" ")
                    except Exception as e:
                        logger.warning(f"ElevenLabs keepalive failed: {e!r}")
        except asyncio.CancelledError:
            pass
