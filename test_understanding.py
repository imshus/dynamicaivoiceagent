"""Does the agent understand a caller who does not talk like the FAQ?

Run it yourself — it spends your OpenAI key (about twenty small calls, no
Deepgram and no ElevenLabs):

    python test_understanding.py            # all of them
    python test_understanding.py gold       # only lines tagged gold
    python test_understanding.py walk       # one guided walkthrough, turn by turn

Each line below is how a jeweller actually says the thing, mid-work, in
Hinglish — never in the words the notes use. The right column says which note
should answer it, so you can read down and see where it goes wrong. The last
few are deliberately outside the notes: those must be declined, not invented.
"""
from __future__ import annotations

import asyncio
import sys

import agent

CASES: list[tuple[str, str, str]] = [
    # tag, what the caller says, the note that should answer
    ("gold", "bhai ye rate kahan se aa raha hai, mujhe apne sarafa wala rate lagana hai",
     "choose your Bullion source"),
    ("gold", "mere dashboard pe sirf bais carat dikh raha hai, chaubis bhi chahiye",
     "choose which rates are shown"),
    ("gold", "jo rate aata hai usme mujhe pachas rupaye jodkar dikhana hai har baar",
     "add or subtract on Masters, Gold, Rates"),
    ("gold", "meri dukan me tunch alag chalti hai, nabbe pe kaam karta hun",
     "edit the purity percentage"),
    ("gold", "customer cash de raha hai, ab konsi rate lagegi",
     "tap RTGS or Cash while calculating"),
    ("diamond", "tag pe packet number likha hota hai, wo apne aap uth jayega kya",
     "packet code in Add Diamond Rates"),
    ("diamond", "ek hi ring me do alag alag diamond lage hain, dono ka hisab hoga",
     "both detected, shown separately"),
    ("diamond", "is customer ko thoda kam rate lagana hai heere ka, ho jayega",
     "edit the rate while calculating"),
    ("colorstone", "ring me ruby lagi hui hai, uska kya hoga",
     "colorstone detected on scanning"),
    ("colorstone", "stone ka rate wala khana khali aa raha hai",
     "type the rate, MRP calculates"),
    ("labour", "hum log majduri tag pe likhte hi nahi",
     "field is blank, enter labour rate"),
    ("labour", "har baar labour type karni padti hai, fix nahi ho sakti",
     "predefine in Masters, Labour Charges"),
    ("invoice", "bill customer ke whatsapp pe bhej sakte hain kya",
     "Generate Invoice, Preview, share"),
    ("invoice", "mujhe pakka wala sarkari bill chalu karwana hai",
     "e-invoice, email info@mrpscan.com"),
    ("wishlist", "abhi ka hisab bacha ke rakhna hai, customer sochkar aayega",
     "Add to Wishlist"),
    ("employee", "naya ladka rakha hai counter pe, usko chalane ki id de do",
     "Add New Employee"),
    ("employee", "wo rate apne aap badal deta hai, ye band karo",
     "uncheck Gold in Rate Edit Access"),
    ("employee", "ladke ne kaam chhod diya, ab uska phone band karna hai",
     "turn off Active Account"),
    ("employee", "usko sirf RTGS dikhe, cash nahi",
     "Gold Rate Options While Calculating"),
    ("outside", "meri GST return bhi file kar dijiye",
     "NOT in the notes — must decline, offer the team"),
    ("outside", "aaj sone ka bhav kya chal raha hai",
     "NOT in the notes — must decline, no invented rate"),
]


# One caller, one job, spoken the way it really goes: a question, a nod, a
# moment of being lost, and a stretch where they say nothing at all because
# they are looking at their screen. None is that silence — on a real call the
# agent checks in by itself after NUDGE_AFTER_SECONDS. The agent should give
# ONE step per turn and hold its place until the caller confirms.
WALK: list[str | None] = [
    "mujhe apne sarafa wala rate lagana hai app me",
    "haan",
    "nahi mil raha, kahan hai wo",
    "haan ab dikh gaya",
    None,                       # the caller goes quiet, doing the step
    "ho gaya",
]


async def walkthrough(client, system: str) -> None:
    history: list[dict] = [{"role": "system", "content": system}]
    for said in WALK:
        if said is None:
            print(f"caller: … says nothing for {agent.NUDGE_AFTER_SECONDS:.0f} seconds")
            messages = history + [{"role": "system", "content": agent.NUDGE_INSTRUCTION}]
        else:
            print(f"caller: {said}")
            history.append({"role": "user", "content": said})
            messages = history
        try:
            r = await client.chat.completions.create(
                model=agent.OPENAI_MODEL, messages=messages, **agent.llm_params())
            reply = (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"!! {e!r}")
            return
        history.append({"role": "assistant", "content": reply})
        print(f"Priya : {agent.clean_for_speech(reply)}\n")
    print("Read down: one step per turn, the same step again where the caller was lost,\n"
          "and a check-in rather than silence when they said nothing.")


async def ask(client, system: str, said: str) -> str:
    try:
        r = await client.chat.completions.create(
            model=agent.OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": said}],
            **agent.llm_params(),
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"!! {e!r}"


async def main() -> None:
    only = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if only == "walk":
        system = agent.build_system_prompt(None, "female")
        print(f"{agent.OPENAI_MODEL}, reasoning {agent.OPENAI_REASONING_EFFORT} — "
              f"{len(WALK)} turns\n")
        await walkthrough(agent.get_llm(), system)
        return
    cases = [c for c in CASES if not only or c[0] == only]
    if not cases:
        print(f"No cases tagged {only!r}. Tags: {sorted({c[0] for c in CASES})}")
        return

    system = agent.build_system_prompt(None, "female")
    print(f"{agent.OPENAI_MODEL}, reasoning {agent.OPENAI_REASONING_EFFORT}, "
          f"system prompt {len(system):,} chars — {len(cases)} calls\n")

    client = agent.get_llm()
    replies = await asyncio.gather(*(ask(client, system, said) for _, said, _ in cases))
    for (tag, said, expected), reply in zip(cases, replies):
        print(f"[{tag}] {said}")
        print(f"   should answer: {expected}")
        print(f"   said: {agent.clean_for_speech(reply)}\n")


if __name__ == "__main__":
    asyncio.run(main())
