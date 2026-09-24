# LLM Cost Autopilot

A router that sits between an app and several AI model providers. Instead of sending every request to one expensive model, it reads the prompt, predicts how complex it is, and sends it to the cheapest model that can handle it — checking the answer's quality afterward and escalating to a stronger model if it falls short.

It now accepts three kinds of input: **text**, **audio**, and **video**. Audio and video are transcribed first, then routed the same way as text.

## Why this exists

Most real-world LLM traffic is simple. Sending "what's 5+5" to the same flagship model as a ten-page analysis wastes money at scale. This project builds the routing layer that fixes that, with an automated check so the savings never come at the cost of a silently worse answer.

## How a request flows

```
Prompt (text, audio, or video)
        |
  [audio/video only] extract speech -> transcript (Groq Whisper)
        |
  Complexity Classifier (Random Forest, 3 tiers)
        |
  Router (tier -> model, config/routing.yaml)
        |
  Model call (Groq / Gemini / local Ollama)
        |
  Quality Verifier (LLM-as-judge, 1-5)
        |
  Score < 4? -> escalate to a stronger model
        |
  SQLite log -> Streamlit dashboard / app
```

**Complexity Classifier** — a Random Forest trained on hand-labeled prompts, using features like keyword signals, prompt length, and output-format complexity to predict Simple / Moderate / Complex.

**Router** — a YAML-configurable tier-to-model mapping, hot-swappable via a `PUT /v1/routing-config` API endpoint without redeploying.

**Verifier** — a stronger model grades every response 1-5. A low score triggers an automatic re-run on a better model, so a cheap routing decision never silently ships a bad answer.

**Audio input** — the recording is sent to Groq's Whisper API, which is billed per minute regardless of which model later answers it. The transcript is then routed and answered like any text prompt.

**Video input (current version)** — the video's soundtrack is extracted locally with ffmpeg and transcribed the same way as audio. This version only understands what is *said* in a video, not what's shown on screen. A visual understanding mode (routing frames to a video-capable model like Gemini) is a planned next step, not yet built.

## Using it

### `app.py` — the interactive app

```
streamlit run app.py
```

Three tabs — **Text**, **Audio**, **Video** — each showing:
- the answer
- which lane the request took (Simple / Moderate / Complex) and which model answered
- a cost breakdown: the model call, the transcription cost (audio/video), and the total
- token counts and a reference comparison against GPT-4o list pricing (GPT-4o is never called — it's a price-only baseline)

### `dashboard.py` — the aggregate view

```
streamlit run dashboard.py
```

Cost totals against the GPT-4o reference, routing distribution, quality score distribution, escalation rate, and a browsable log of recent requests with their full answers.

### API

```
uvicorn api:app --reload
```

Exposes the same pipeline over HTTP for text requests. See `api.py` for the current endpoints.

## Project layout

```
core/
  models.py        standardized response object (text, tokens, cost, latency)
  registry.py       model configs: provider, cost/token, quality tier
  interface.py      unified send_request() across Groq / Gemini / Ollama
  complexity.py     feature extraction for the classifier
  router.py         tier prediction + model lookup
  verifier.py       LLM-as-judge scoring + escalation
  logger.py         SQLite read/write
  pipeline.py       end-to-end orchestration
  audio.py          Groq Whisper transcription + cost estimate
  video.py          ffmpeg audio extraction from video, then transcription

config/
  routing.yaml      tier -> model mapping (hot-editable)

data/
  labeled_prompts.csv        classifier training data
  complexity_classifier.pkl  trained model
  autopilot.db                logged requests

app.py            interactive text/audio/video interface
dashboard.py      aggregate cost & quality dashboard
api.py            FastAPI service
Dockerfile / docker-compose.yml
requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in your keys
```

`.env` needs:
```
GROQ_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434
```

Audio/video transcription also needs `imageio-ffmpeg` (in `requirements.txt`) — it ships its own ffmpeg binary, so no separate install is required.

## Honest limitations, as of this version

- **Judge cost isn't logged.** The verifier's own model call has a real cost that isn't currently written to the database, so any cost-savings percentage should be treated as a lower bound until that's fixed.
- **No provider fallback yet.** If a provider hits its rate limit mid-request, the request fails rather than retrying on a fallback model, even though `routing.yaml` defines fallback models.
- **Video is audio-only for now.** A silent video, or one where the answer depends on what's on screen, won't work yet.
- **Free-tier quotas are real.** Groq and Gemini's free tiers cap out well below production volume; expect 429s under sustained testing.

## Cost results

*(Fill this in after a clean run once the judge-cost logging fix above is in place — the current logged numbers understate total cost.)*

| Metric | Value |
|---|---|
| Requests | — |
| Total cost (incl. judge) | — |
| Reference cost (all-GPT-4o) | — |
| Cost reduction | — |
| Average quality score | — |
| Escalation rate | — |

## Tech stack

Python, FastAPI, scikit-learn, SQLite, Streamlit, Docker, Groq API (LLM + Whisper), Gemini API, Ollama (local fallback), ffmpeg (via `imageio-ffmpeg`).