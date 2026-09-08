"""Ghana Twi Voice -- a spoken Twi conversation, served from Modal.

One turn:

    mic audio (webm/opus)
      -> ffmpeg                 -> 16 kHz mono
      -> griot-nano-1 + KenLM   -> a Twi transcript          Conformer-CTC, on CPU
      -> Gemini 2.5 Flash       -> {understood, reply}       reads a noisy transcript, answers in Twi
      -> stable-twi-tts (ONNX)  -> 22.05 kHz speech          voice twi-6
      -> libmp3lame             -> one MP3 per sentence, pushed over the websocket as it is ready

Recognition is a dedicated Ghanaian model rather than Gemini's own audio understanding, because
that does not work for Twi: given real Twi speech Gemini transcribed it as the English sentence
"Why won't you come out?" and confidently answered that instead. See asr.py.

Nothing here needs a GPU. The recogniser runs at 0.09-0.19x realtime on CPU threads and the
voice is ONNX, so the whole app is a CPU container -- which also keeps the cold start in
seconds rather than the ~40 s the earlier Wav2Lip version paid for CUDA and torch (archive/).

Deploy:
    modal deploy modal_app.py
"""
from __future__ import annotations

import modal

APP_NAME = "ghana-twi-voice"

GEMINI_MODEL = "gemini-2.5-flash"
# Lowest measured Twi-only phoneme error in the published set (27%, tied with twi-7).
TTS_VOICE = "twi-6"
GRIOT_DIR = "/opt/griot"
LM_BIN = "/opt/lm/multilingual.bin"
# Modal caps a single websocket message at 2 MiB. At 64 kbps that is over four minutes of
# audio and a clause is a few seconds, so this is a guard rail rather than a real limit.
MAX_WS_BYTES = 1_800_000
# The recogniser's transcript is the primary evidence; the recording goes along with it so the
# model can still hear a question's intonation, a name or a number that the transcript mangled.
SEND_AUDIO_TO_GEMINI = True

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        # kenlm ships no wheel for any version on any platform, so it compiles here. cmake and
        # a compiler build the Python extension; the boost components are what its CMake needs
        # for the `build_binary` and `lmplz` executables, and `build_binary` is the whole reason
        # to compile it rather than live with the ARPA -- see the LM conversion below.
        "build-essential", "cmake",
        "libboost-program-options-dev", "libboost-system-dev",
        "libboost-thread-dev", "libboost-test-dev",
        "zlib1g-dev", "libbz2-dev", "liblzma-dev",
    )
    .env({
        "XDG_CACHE_HOME": "/opt/cache",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        # kenlm 0.2.0's CMakeLists declares a pre-3.5 minimum, and CMake 4 -- which is what
        # this base image ships -- removed compatibility with those outright, so the build dies
        # at the first line of configuration. This is the escape hatch CMake provides for
        # exactly that case. It fails only in the image: a distro with CMake 3.x builds kenlm
        # without it, which is why this looked like a missing dependency at first.
        "CMAKE_POLICY_VERSION_MINIMUM": "3.5",
    })
    # CPU-only torch, from PyTorch's cpu index: the default PyPI wheel drags in ~2 GB of CUDA
    # that nothing in this app can use.
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.14.1",
        "soundfile==0.12.1",
        "safetensors==0.4.5",
        "tokenizers==0.20.3",
        "kenlm==0.2.0",
        "pyctcdecode==0.5.0",
        "huggingface_hub==0.26.2",
        "stable-twi-tts[twi]==0.2.1",
        "ghana-g2p==0.1.1",
        # Pinned, and the pin is load-bearing. ghana-g2p asks only for `africa-g2p>=0.1.1`, so
        # a fresh build picks up 0.2.0, which writes affricates with a combining tie bar --
        # 'd͡ʑ' (U+0361) where the voice's symbol table has 'dʑ', and 't͡ɕʰ' where it has 'tɕʰ'.
        # Those are the sounds Twi spells `gy` and `ky`, so with 0.2.0 installed almost every
        # real Twi sentence dies in the front-end with PhonemeError and there is nothing to
        # say. 0.1.1 is the inventory the voice was trained on.
        "africa-g2p==0.1.1",
        "google-genai==2.22.0",
        "fastapi==0.115.6",
    )
    # Bake the voice in so a cold container has nothing to download before it can speak.
    .run_commands("python -c 'from stable_twi_tts import StableTwiTTS;"
                  " StableTwiTTS.from_pretrained()'")
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download;"
        f" snapshot_download('Qlerqly/griot-nano-1', local_dir='{GRIOT_DIR}')\"",
    )
    # The language model ships as a prebuilt binary trie rather than being converted here.
    # `pip install kenlm` builds the Python extension but not the `build_binary` executable
    # (it configures with -DBUILD_PYTHON_STANDALONE=ON), so there is nothing in the image to do
    # the conversion with. The binary is worth having anyway: 80 MB against the published
    # 225 MB ARPA, and it mmaps in under a second where digesting the ARPA costs 15-25 s of
    # every container's startup.
    #
    # To regenerate it, on a machine whose CMake is older than 4 so kenlm's executables build:
    #     pip install kenlm==0.2.0
    #     hf download Qlerqly/griot-nano-1-kenlm --local-dir lm
    #     build_binary trie lm/multilingual.arpa assets/multilingual.bin
    .add_local_file("assets/multilingual.bin", LM_BIN, copy=True)
    .add_local_python_source("asr")
)

# FastAPI resolves endpoint annotations against the *module* globals, and this file uses
# `from __future__ import annotations`, so every annotation is a string by the time FastAPI
# looks at it. Importing FastAPI inside the method that builds the app leaves it unable to
# resolve `sock: WebSocket`, and an unresolvable annotation is treated as a query parameter:
# every websocket connect then closes with 1008 "Field required: query.sock", which reaches the
# browser as an opaque HTTP 500. Hoisting these into module scope is what makes them resolvable.
with image.imports():
    from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware

app = modal.App(APP_NAME)
gemini_secret = modal.Secret.from_name("gemini-api-key")

SYSTEM_PROMPT = """\
You are a warm, plain-spoken Ghanaian woman having a spoken conversation in Twi. Someone has \
just said something to you out loud, in Twi.

You are given a machine transcript of what they said, and usually the recording as well. The \
transcript comes from a Ghanaian speech recogniser, and Akan is its weakest language: roughly \
two words in five come out wrong. It confuses ɛ with e, a and i, and ɔ with o and u; it drops \
the final consonant of a word; and it splits and joins words in the wrong places. So it \
usually produces something that is *nearly* a Twi phrase, spelled badly.

Your job is to read through the bad spelling to the ordinary Twi sentence underneath. Almost \
everything people say to you is everyday conversational Twi -- a greeting, how-are-you, thanks, \
a short question about you or about Ghana. When a transcript is close to a common phrase, that \
common phrase is what they said. Read it aloud in your head and let the sound guide you, the \
way you would understand someone on a bad phone line.

Worked examples of this recogniser's output and what was actually said:
  "wo te sɛ ana paɛ"        -> "Wo ho te sɛn?"
  "me wɔyɛnko"              -> "Woayɛ ankonam anaa?"
  "mmienu ne duonunsia"     -> "Mmienu ne aduonu nsia"
  "medase pa"               -> "Medaase paa"
  "wo din de sɛ"            -> "Wo din de sɛn?"

Default to answering. If several readings are possible, take the most ordinary one and reply to \
that -- a natural reply to a likely reading is far more useful to the person than being asked to \
repeat themselves. You do not need to be certain; you only need a sensible guess.

Ask them to repeat only when the transcript is genuinely empty or pure noise with no Twi in it \
at all. That should be rare. Refusing to engage with a readable greeting is the worst thing you \
can do here.

Then reply, obeying all of these:
- Reply in Twi ONLY. If they seem to have spoken English, or you cannot tell what language it \
was, still reply in Twi.
- Use proper Twi orthography, including the letters ɛ and ɔ where they belong.
- Write out every number as Twi words: "aduonu num", never "25". A digit in your reply is \
skipped silently by the voice that speaks it, so "me wɔ mfe 25" would be spoken as \
"me wɔ mfe" -- the wrong sentence, with nothing to show anything went missing.
- Avoid English words. The voice can pronounce them, but its English is noticeably poorer than \
its Twi, so a borrowed word is the worst-sounding part of any sentence. If a concept has no \
everyday Twi word, describe it in plain Twi instead.
- At most 2 sentences, and keep them short. Your reply is going to be spoken aloud, so write it \
the way someone would actually say it, not the way it would be written down.
- No emoji, no markdown, no bullet points, no parentheses, no quotation marks.

Return JSON with three fields:
  "understood": what you believe they actually said, written in clean Twi. This is shown back \
to them so they can check whether you heard them right, so make it a faithful reconstruction of \
their words, not a paraphrase and not your answer. Leave it as an empty string ONLY if you are \
setting confident to false.
  "confident": true if you are reasonably sure of "understood"; false if you are really only \
guessing. When this is false your reply should say in Twi that you did not catch it and ask them \
to repeat -- and it must be false in that case, so the two never disagree. Never write a \
confident reconstruction and then claim you did not understand.
  "reply": your Twi reply.
"""

REPLY_SCHEMA = {
    "type": "OBJECT",
    "properties": {"understood": {"type": "STRING"},
                   "confident": {"type": "BOOLEAN"},
                   "reply": {"type": "STRING"}},
    "required": ["understood", "confident", "reply"],
}

# Asking the model to rewrite its own digits beats spelling them out here: Twi numerals inflect
# in ways this file has no business guessing at, and a wrong number read confidently is worse
# than the silent drop it was meant to fix.
DESPELL_PROMPT = """\
Rewrite this Twi sentence with every number written out in full as Twi words instead of digits. \
Change nothing else at all -- same meaning, same wording, same punctuation. Reply with only the \
rewritten sentence, nothing else.

"""


class WebsocketCloseShim:
    """Coerce `websocket.close` fields to the types Modal's protobuf bridge accepts.

    Modal serialises outgoing ASGI messages straight into protobuf, passing `reason` into a
    string field and `code` into a uint32 without checking either. Starlette sends a validation
    failure's `reason` as a list of error dicts, which raises `TypeError: bad argument type for
    built-in operation` inside Modal's own serialiser -- killing the connection before the
    handshake finishes, so the browser sees an opaque HTTP 500 with nothing of ours in the
    traceback. Normalising the two fields makes every close frame serialisable, which is also
    what makes the real close code visible instead of a 500.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "websocket":
            return await self.app(scope, receive, send)

        async def guarded(message):
            if message.get("type") == "websocket.close":
                code, reason = message.get("code"), message.get("reason")
                if not isinstance(code, int) or isinstance(code, bool) or not isinstance(reason, str):
                    print(f"normalising unserialisable close frame: {message!r}")
                message = dict(message)
                message["code"] = code if isinstance(code, int) and not isinstance(code, bool) else 1000
                message["reason"] = reason if isinstance(reason, str) else ""
            await send(message)

        return await self.app(scope, receive, guarded)


@app.cls(
    image=image,
    secrets=[gemini_secret],
    # The Conformer is the only heavy thing here and it is happiest with real threads.
    cpu=8.0,
    memory=8192,
    # A CPU container is cheap enough to keep one resident, which removes the cold start from
    # the first thing anyone says -- the worst place to spend it.
    min_containers=1,
    max_containers=6,
    scaledown_window=300,
    timeout=3600,
)
@modal.concurrent(max_inputs=8)
class TwiVoice:
    @modal.enter()
    def load(self):
        import os
        import threading
        import time

        import numpy as np
        from google import genai
        from stable_twi_tts import StableTwiTTS

        import asr

        t = time.time()
        self.griot = asr.Griot(GRIOT_DIR, LM_BIN, threads=8)
        # 4 ONNX threads is plenty for the voice: a two-sentence reply is well under a second
        # of CPU, and the recogniser wants the rest.
        self.tts = StableTwiTTS.from_pretrained(quiet=True, num_threads=4)
        self.gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        # One lock each: the two models are separate processes' worth of state and a turn can
        # be recognising for one caller while synthesising for another.
        self.asr_lock = threading.Lock()
        self.tts_lock = threading.Lock()

        # Warm both graphs so the first real request does not pay for lazy initialisation.
        self.griot.transcribe(np.zeros(asr.SAMPLE_RATE, dtype=np.float32))
        with self.tts_lock:
            self.tts.synthesize("Akwaaba.", voice=TTS_VOICE, language="twi")

        self.warm_seconds = round(time.time() - t, 1)
        print(f"ready in {self.warm_seconds}s -- griot-nano-1 + KenLM, voice {TTS_VOICE}")

    # ---------------------------------------------------------------- pipeline stages

    def _decode_mic(self, blob: bytes) -> tuple["object", bytes]:
        """Whatever the browser recorded -> (float32 mono at 16 kHz, the same as wav bytes).

        MediaRecorder gives webm/opus on Chrome and mp4/aac on Safari. The array feeds the
        recogniser; the wav feeds Gemini, which accepts neither of those containers reliably.
        """
        import io
        import subprocess

        import numpy as np
        import soundfile as sf

        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
             "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
            input=blob, capture_output=True)
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError(f"could not decode the recording: {p.stderr.decode()[:300]}")
        audio, sr = sf.read(io.BytesIO(p.stdout), dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), p.stdout

    def listen(self, audio) -> str:
        with self.asr_lock:
            return self.griot.transcribe(audio)

    def _ask_gemini(self, transcript: str, wav_bytes: bytes | None,
                    history: list[dict]) -> dict:
        """A noisy transcript (and the recording) in, {understood, reply} out.

        Thinking is switched off: this is a voice loop, and the budget would show up as silence
        with nothing on screen to explain it.
        """
        import json

        from google.genai import types

        contents = []
        for turn in history[-6:]:
            contents.append(types.Content(role="user",
                                          parts=[types.Part.from_text(text=turn["user"])]))
            contents.append(types.Content(role="model",
                                          parts=[types.Part.from_text(text=turn["reply"])]))

        parts = [types.Part.from_text(
            text=f"Machine transcript of what they just said:\n{transcript or '(nothing)'}")]
        if wav_bytes and SEND_AUDIO_TO_GEMINI:
            parts.append(types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"))
        contents.append(types.Content(role="user", parts=parts))

        resp = self.gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=REPLY_SCHEMA,
                temperature=0.8,
                max_output_tokens=500,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        out = json.loads(resp.text)
        return {"understood": (out.get("understood") or "").strip(),
                # Defaults to true only when the field is genuinely absent; an explicit false
                # has to survive, since it is what stops the page claiming "you said X" beside
                # a reply that says it could not make out a word.
                "confident": bool(out.get("confident", True)),
                "reply": (out.get("reply") or "").strip()}

    def _spell_numbers(self, reply: str) -> str:
        """Replace digits with Twi words, by asking the model that wrote them.

        The prompt already forbids digits, but a prompt is a soft guard and the failure it
        guards against is silent: the Twi front-end drops digits without raising, so
        "me wɔ mfe 25" is spoken as "me wɔ mfe" -- a different, wrong sentence that nothing in
        the pipeline flags. Verified against the deployed voice, not assumed.
        """
        import re

        from google.genai import types

        if not re.search(r"\d", reply):
            return reply
        print(f"reply contains digits, asking for a rewrite: {reply!r}")
        try:
            resp = self.gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=DESPELL_PROMPT + reply,
                config=types.GenerateContentConfig(
                    temperature=0.0, max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)),
            )
            fixed = (resp.text or "").strip()
        except Exception as exc:
            print(f"digit rewrite failed ({type(exc).__name__}: {exc})")
            fixed = ""
        if not fixed or re.search(r"\d", fixed):
            # Better a sentence missing its number than one that states the wrong number.
            return re.sub(r"\s*\d+\s*", " ", fixed or reply).strip()
        return fixed

    def _sentences(self, text: str) -> list[str]:
        """Split the reply so the first clause can start playing while the second is still made."""
        from stable_twi_tts.g2p import split_sentences

        parts = [s.strip() for s in split_sentences(text) if s.strip()]
        return parts or ([text.strip()] if text.strip() else [])

    def clip(self, text: str, voice: str = TTS_VOICE) -> tuple[bytes, float, dict]:
        """One clause -> (mp3 bytes, duration in seconds, per-stage timings).

        MP3 rather than the raw wav the voice produces: about five times smaller for the same
        clause, which is the difference that shows up as time-to-first-audio on a phone.
        """
        import subprocess
        import time

        import numpy as np

        t0 = time.time()
        with self.tts_lock:
            s = self.tts.synthesize(text, voice=voice, language="twi")
        t1 = time.time()

        pcm = np.clip(np.asarray(s.audio, dtype=np.float32), -1.0, 1.0)
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "f32le", "-ar", str(s.sample_rate), "-ac", "1", "-i", "pipe:0",
             "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"],
            input=pcm.tobytes(), capture_output=True)
        if p.returncode != 0 or not p.stdout:
            raise RuntimeError(f"mp3 encode failed: {p.stderr.decode()[:300]}")
        t2 = time.time()

        duration = len(s.audio) / s.sample_rate
        return p.stdout, duration, {
            "tts": round(t1 - t0, 2), "encode": round(t2 - t1, 2),
            "total": round(t2 - t0, 2),
            "realtime_ratio": round((t2 - t0) / max(duration, 1e-6), 2)}

    # ---------------------------------------------------------------------- web surface

    @modal.asgi_app()
    def web(self):
        import asyncio
        import json
        import time
        import traceback

        api = FastAPI(title="Ghana Twi Voice")
        # The frontend is a static Hugging Face Space on a different origin, so the browser
        # preflights everything here.
        api.add_middleware(
            CORSMiddleware,
            allow_origin_regex=r"https://.*\.hf\.space|https://huggingface\.co"
                               r"|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @api.get("/health")
        def health():
            import asr

            return {"ok": True, "voice": TTS_VOICE, "model": GEMINI_MODEL,
                    "asr": "griot-nano-1", "kenlm": self.griot.decoder is not None,
                    "beam_width": asr.BEAM_WIDTH, "warm": self.warm_seconds}

        @api.get("/say")
        async def say(text: str):
            """Text -> one MP3. No mic, no ASR, no Gemini; isolates the voice when debugging."""
            data, dur, timings = await asyncio.to_thread(self.clip, text[:400])
            print(f"/say {timings} for {dur:.2f}s of audio")
            return Response(data, media_type="audio/mpeg", headers={
                "X-Timings": json.dumps(timings), "X-Duration": f"{dur:.3f}"})

        @api.websocket("/ws")
        async def ws(sock: WebSocket):
            await sock.accept()
            history: list[dict] = []      # per connection, so callers never share context
            current_voice: str = TTS_VOICE  # set by an explicit {"type":"voice"} from the page

            async def speak(reply: str) -> int:
                sent = 0
                for i, sentence in enumerate(self._sentences(reply)):
                    try:
                        data, dur, timings = await asyncio.to_thread(
                            self.clip, sentence, current_voice)
                    except Exception as exc:
                        # A clause the Twi front-end cannot pronounce should cost that clause,
                        # not the rest of the reply.
                        traceback.print_exc()
                        await sock.send_text(json.dumps(
                            {"type": "warning", "index": i,
                             "detail": f"{type(exc).__name__}: {exc}"[:200]}))
                        continue
                    if len(data) > MAX_WS_BYTES:
                        await sock.send_text(json.dumps(
                            {"type": "warning", "index": i,
                             "detail": f"clause too large to send ({len(data)} bytes)"}))
                        continue
                    print(f"clause {i} {timings} for {dur:.2f}s of audio")
                    await sock.send_text(json.dumps(
                        {"type": "audio", "index": i, "text": sentence,
                         "duration": round(dur, 3), "bytes": len(data), "timings": timings}))
                    await sock.send_bytes(data)
                    sent += 1
                return sent

            try:
                while True:
                    msg = await sock.receive()
                    if msg["type"] == "websocket.disconnect":
                        break

                    if msg.get("bytes"):
                        t0 = time.time()
                        await sock.send_text(json.dumps({"type": "status", "stage": "hearing"}))
                        audio, wav = await asyncio.to_thread(self._decode_mic, msg["bytes"])
                        transcript = await asyncio.to_thread(self.listen, audio)
                        t1 = time.time()
                        # The raw recogniser output goes to the page as well as to Gemini: when
                        # a turn goes wrong it is the only way to see whether the recogniser or
                        # the model was at fault.
                        await sock.send_text(json.dumps(
                            {"type": "transcript", "text": transcript,
                             "seconds": round(t1 - t0, 2)}))

                        await sock.send_text(json.dumps({"type": "status", "stage": "thinking"}))
                        turn = await asyncio.to_thread(self._ask_gemini, transcript, wav, history)
                        reply = await asyncio.to_thread(self._spell_numbers, turn["reply"])
                        print(f"asr={transcript!r} understood={turn['understood']!r} "
                              f"confident={turn['confident']} reply={reply!r}")
                        await sock.send_text(json.dumps(
                            {"type": "understood", "text": turn["understood"],
                             "confident": turn["confident"]}))
                        await sock.send_text(json.dumps({"type": "reply", "text": reply}))
                        if not reply:
                            await sock.send_text(json.dumps({"type": "done", "clauses": 0}))
                            continue
                        history.append({"user": turn["understood"] or transcript or "(audio)",
                                        "reply": reply})

                    elif msg.get("text"):
                        req = json.loads(msg["text"])
                        if req.get("type") == "ping":
                            await sock.send_text(json.dumps({"type": "pong"}))
                            continue
                        if req.get("type") == "voice":
                            # Pick the voice for subsequent replies. Falls silently back to the
                            # default if the name is unknown, so a stale dropdown can never take
                            # the conversation down.
                            if isinstance(req.get("name"), str) and req["name"] in {
                                v.name for v in self.tts.voices.by_language("twi")}:
                                current_voice = req["name"]
                            continue
                        # "say" speaks given text directly, bypassing mic, ASR and Gemini.
                        reply = (req.get("text") or "").strip()[:400]
                        if not reply:
                            continue
                        await sock.send_text(json.dumps({"type": "reply", "text": reply}))
                    else:
                        continue

                    await sock.send_text(json.dumps(
                        {"type": "done", "clauses": await speak(reply)}))

            except WebSocketDisconnect:
                pass
            except Exception as exc:
                traceback.print_exc()
                try:
                    await sock.send_text(json.dumps(
                        {"type": "error", "detail": f"{type(exc).__name__}: {exc}"[:300]}))
                except Exception:
                    pass

        return WebsocketCloseShim(api)
