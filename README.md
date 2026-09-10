# Dynamic Voice Agent

Talk to an AI agent from the browser. Every reply is written live by the model — nothing is scripted.

**Browser mic → Deepgram Flux (STT + turn-taking) → GPT-5.6 Luna (streamed) → ElevenLabs (streamed TTS) → browser speaker**

## Run

```bash
pip install -r requirements.txt
copy .env.example .env      # fill in DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY
python server.py
```

Open **http://localhost:8100**, click **Start call**, and talk. Use headphones.

## Change the agent

- `prompt.md` — the persona. Edits apply on the next call, no restart.
- `faq.md` — **what the agent knows**: the MRPscan FAQ, every question and answer in English and Hindi. It is appended to the system prompt on every call, so answers come from approved text instead of invention. Add a question by adding it to this file; the next caller gets it, no restart and no code change. Set `KNOWLEDGE_FILE` to another file, or empty, to change or drop it.
- The textarea on the page overrides `prompt.md` for that one call — paste a business description and talk to that agent. The FAQ still gets attached; empty `KNOWLEDGE_FILE` for an agent that should not know it.

## Understanding the caller

A jeweller does not ask the FAQ's question. They say "bhai rate kahan se aa raha hai" and mean *choose your Bullion source*. Three things carry that:

1. **Deepgram keyterms** (`DEEPGRAM_KEYTERMS`) bias the words this helpline turns on — colorstone, karat, RTGS, tunch, packet code — so Flux stops hearing "cash tone" for "colorstone". A word that arrives wrong can never be understood, so this comes before any prompt wording.
2. **`prompt.md` asks for intent, not matching**: work out what the caller wants, answer with the note that solves it, and ask one short question when two notes genuinely both fit.
3. **`OPENAI_REASONING_EFFORT=low`** gives the model a moment to pick the right note out of thirty. On a turn where Flux fires `EagerEndOfTurn`, that thinking happens while the caller is still finishing, so it usually costs nothing. Set it back to `none` if you want the last few milliseconds.

`python test_understanding.py` reads twenty-one lines the way a jeweller really says them, past each one to the model, and prints what came back next to the note that should have answered it. It spends only your OpenAI key. Add a tag to run one area: `python test_understanding.py employee`.

## One step at a time

A caller cannot follow four menu steps read out in one breath. When the answer is a path, the agent says the first step, asks whether they are there, and stops. It gives the next step only once they say they have done it, and if they say they cannot find it, it stays on that step and describes it differently instead of moving on. A caller who says they know the app, or asks for all of it, gets it all at once.

If the caller does not confirm, the agent stays where it is: it asks again whether that step is done, and describes the screen another way when they sound stuck. It never gives two steps in one breath.

Waiting for the caller changes three things in the engine:

- **Silence gets an answer, not more silence.** A caller doing the step says nothing, and the agent used to wait mutely until the line timed out. After `NUDGE_AFTER_SECONDS` (12) it checks in once — "have you opened it?" — in whatever language the call is in, and at most `NUDGE_MAX` (2) times before leaving them alone. The check-in is a note to the model, never stored in the conversation, so it cannot pile up. Their next word resets the count.

- **A nod is a turn, not an interruption.** "Haan" said on top of the step is too small to cut the agent off, but it is not the agent's own echo either, so it is held and answered the instant the line goes quiet. Before this it was discarded and the caller had to say it twice. Genuine echo — the agent's own words coming back off a speaker — is still dropped.
- **Silence is now expected.** The caller goes quiet while they tap through their screens, so the MRPscan app's silence hangup moved from ten seconds to forty-five (`PRATHAM_AI_SILENCE_MS`).

`python test_understanding.py walk` plays one whole walkthrough — a question, two nods, and a turn where the caller is lost — and prints it turn by turn.
- **Female / Male** switch on the page — picks the voice (`ELEVENLABS_VOICE_ID_FEMALE` / `_MALE`), the name (`AGENT_NAME_*`) and the greeting, and tells the model which gender it speaks as (Hindi verbs are gendered, so voice and words must agree).
- `GREETING` in `.env` — the first thing the agent says; `{name}` is filled in. `GREETING_FEMALE` / `GREETING_MALE` override it per gender. Leave empty for no greeting.

## Voice: one tone for the whole call

- **One ElevenLabs stream per call.** The greeting and every reply are text appended to the same input-streaming generation, kept alive through silences. A fresh generation per reply is what makes the voice come back on a slightly different tone, pace or level; one stream cannot.
- `stability=1.0`, `style=0`, speaker boost off, `speed=1.0` fixed.
- Exclamation marks are turned into full stops before speech ("Hello!" is what makes the voice jump bright and then settle), and the prompt asks for a calm, even tone.
- On barge-in, audio for the cancelled text is dropped by character position (ElevenLabs' alignment data), so the same stream carries straight on with the next reply.

## Socket details

- Public URL: **https://prathamai.mrpscan.com** (`PRATHAM_AI_URL` in `.env`)
- Socket: **wss://prathamai.mrpscan.com/ws**
- `GET /prompt` returns the default instructions, gender, names and `socket_url`; `GET /health` returns `{"status":"ok"}`.

Any client can drive a call over that socket:

1. Open the socket and send `{"type": "start", "instructions": "<optional, replaces prompt.md>", "gender": "female" | "male"}`.
2. The server answers `{"type": "ready", "sample_rate": 24000, "gender": ..., "name": ...}` and speaks the greeting.
3. Send the microphone as **binary frames: 16 kHz, mono, 16-bit little-endian PCM** (20–50 ms per frame works well).
4. Receive **binary frames: 24 kHz, mono, 16-bit PCM** to play back, plus JSON text frames: `user` (what was heard), `agent` (`text`, `final`, `interrupted`), `retract` (drop the last question), `clear` (stop playback now), `error`.
5. Send `{"type": "interrupt", "reason": "<optional>"}` the moment your own user starts talking over the agent: the reply is cancelled, its unheard text retracted and queued audio dropped — the same path Deepgram's `StartOfTurn` takes. A phone hears the agent through its own speaker, so the client knows first and waiting for the turn event costs a beat.
6. Send `{"type": "stop"}` or close the socket to end the call.

**Echo on a speakerphone.** The microphone stays open for the whole call, so the agent hears itself. A client watches its own microphone level while the agent speaks — the level settles at whatever the speaker feeds back, and a voice clearly above that is the caller — then stops playback and sends `interrupt`. The server guards the same case from its side: a transcript arriving while the agent talks counts as its own echo when most of its words are ones the agent just said (`BARGE_IN_ECHO_OVERLAP`, default 0.6, matched against the current reply and the one before it), or when it carries fewer than `BARGE_IN_MIN_NEW_WORDS` (default 2) words the agent did not say. `BARGE_IN_MIN_CHARS` (default 6) drops the shortest fragments.

The page at `/` does exactly this. When the page is hosted on any non-localhost origin it connects to `socket_url`; on localhost it uses its own server.

## Host it (prathamai.mrpscan.com)

On an Ubuntu box with Caddy installed and DNS pointing `prathamai.mrpscan.com` at it:

```bash
git clone <this repo> /home/ubuntu/pratham-ai && cd /home/ubuntu/pratham-ai
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # keys, voice IDs, PRATHAM_AI_URL=https://prathamai.mrpscan.com
sudo cp deploy/pratham-ai.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now pratham-ai
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

Check `https://prathamai.mrpscan.com/health`, then open the site and click Start call. Ports 80 and 443 must be open; the app itself listens only on `127.0.0.1:8100` behind Caddy.

## MRPscan app (Pratham AI tab)

The MRPscan app's bottom-bar **Pratham AI** tab dials this server directly over the socket above (no browser): `frontend/utils/prathamAiCall.ts` streams the phone mic up as 16 kHz PCM and plays the 24 kHz replies through `react-native-audio-api`. The address it uses, in order:

1. `PRATHAM_AI_URL` on the MRPscan backend, served at `GET /api/v1/app-config` — change it there and restart the backend; no APK rebuild.
2. `EXPO_PUBLIC_PRATHAM_AI_URL` in `frontend/.env` — the build-time fallback (currently `https://prathamai.mrpscan.com`).

## Files

- `server.py` — FastAPI: serves the page, bridges the browser WebSocket to a session.
- `agent.py` — the engine: Flux turn events, speculative LLM drafting, barge-in with retraction, ElevenLabs input-streaming.
- `ui/index.html` — mic capture (16 kHz PCM), playback (24 kHz PCM), live transcript.

## How a turn flows

1. Flux `EagerEndOfTurn` → the model starts drafting the reply while the caller finishes.
2. Flux `EndOfTurn` → if the final transcript matches the draft, it is reused; otherwise a fresh reply starts.
3. Tokens are cut at sentence boundaries (clause boundary for the first piece) and pushed into ElevenLabs' input-streaming socket; audio streams straight to the browser.
4. Flux `StartOfTurn` while the agent is talking → barge-in: playback cleared, reply cancelled. If no audio had played yet, the caller's text is kept and merged with what they say next.

Latency per turn is logged: `first token`, `first audio`, `reply complete`, all measured from end of turn.
