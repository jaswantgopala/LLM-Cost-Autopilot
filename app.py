import html
import os
import tempfile
import time

import streamlit as st
import yaml

from core.audio import transcribe_audio
from core.pipeline import process_request
from core.video import transcribe_video

REF_IN = 2.50 / 1_000_000    # GPT-4o reference price, never called
REF_OUT = 10.00 / 1_000_000

# tier -> (lane name, lane color)
TIERS = {
    1: ("Simple", "#4FD1A5"),
    2: ("Moderate", "#F5B544"),
    3: ("Complex", "#F0688A"),
}

st.set_page_config(page_title="Autopilot", layout="centered")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;700;800&family=Figtree:wght@400;500;600&display=swap');
:root{--bg:#0D1B2A;--panel:#14273B;--line:#2A4763;--text:#E8EEF4;--muted:#8DA2B6;}
.stApp{background:var(--bg);color:var(--text);font-family:'Figtree',system-ui,sans-serif;}
[data-testid="stHeader"]{background:transparent;}
.block-container{max-width:860px;padding-top:3.2rem;padding-bottom:5rem;}
.brand{display:flex;align-items:center;gap:16px;}
.brand h1{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-weight:800;font-size:3.1rem;letter-spacing:-0.03em;margin:0;padding:0;line-height:1;color:var(--text);}
.mark{display:flex;flex-direction:column;gap:5px;width:40px;}
.mark i{display:block;height:6px;border-radius:3px;}
.mark i:nth-child(1){width:100%;background:#4FD1A5;}
.mark i:nth-child(2){width:72%;background:#F5B544;}
.mark i:nth-child(3){width:44%;background:#F0688A;}
.tagline{color:var(--muted);font-size:1.08rem;max-width:56ch;margin:14px 0 30px 0;line-height:1.55;}
.stApp h3{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-weight:700;font-size:1.35rem;letter-spacing:-0.01em;margin-top:2.2rem;color:var(--text);}
[data-testid="stWidgetLabel"] p,label{color:var(--muted);}
.stTabs [data-baseweb="tab-list"]{gap:1.8rem;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{background:transparent;height:auto;padding:0.5rem 0;}
.stTabs [data-baseweb="tab"] p{font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:1.1rem;font-weight:700;color:var(--muted);}
.stTabs [aria-selected="true"] p{color:var(--text);}
.stTabs [data-baseweb="tab-highlight"]{background:var(--text);height:2px;}
.stTabs [data-baseweb="tab-border"]{background:transparent;}
div[data-baseweb="textarea"],div[data-baseweb="base-input"],div[data-baseweb="input"]{background:var(--panel) !important;border-color:var(--line) !important;border-radius:10px !important;}
textarea,input{color:var(--text) !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel);border:1px dashed var(--line);border-radius:10px;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:10px;background:transparent;}
.stButton > button{background:var(--text);color:var(--bg);border:0;border-radius:10px;font-weight:600;padding:0.55rem 1.5rem;}
.stButton > button p{color:var(--bg) !important;font-weight:600;}
.stButton > button:hover{background:#FFFFFF;}
.stButton > button:focus-visible{outline:2px solid #FFFFFF;outline-offset:2px;}
.rule{height:4px;width:72px;background:var(--c);border-radius:2px;margin:30px 0 0 0;}
.route-note{color:var(--muted);margin:0 0 14px 0;}
.lane{display:grid;grid-template-columns:210px 1fr 120px;align-items:center;gap:18px;padding:14px 0;border-top:1px solid var(--line);opacity:0.5;}
.lane:last-child{border-bottom:1px solid var(--line);}
.lane.taken{opacity:1;}
.lname b{display:block;font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:1.1rem;font-weight:700;}
.lane.taken .lname b{color:var(--c);}
.lname span{color:var(--muted);font-size:0.9rem;}
.track{position:relative;height:2px;background:var(--line);}
.track i{position:absolute;left:0;top:0;height:2px;width:0;background:var(--c);}
.lane.taken .track i{width:100%;animation:draw 0.9s ease-out;}
.lane.taken .track::after{content:"";position:absolute;right:-5px;top:-4px;width:10px;height:10px;border-radius:50%;background:var(--c);}
.lend{font-size:0.92rem;color:var(--c);font-weight:600;}
@keyframes draw{from{width:0;}to{width:100%;}}
.lrow{display:flex;justify-content:space-between;gap:16px;padding:10px 0;border-top:1px solid var(--line);color:var(--muted);}
.lrow b{color:var(--text);font-weight:600;font-variant-numeric:tabular-nums;}
.lrow.total{color:var(--text);font-family:'Bricolage Grotesque',system-ui,sans-serif;font-size:1.25rem;font-weight:700;border-bottom:1px solid var(--line);}
.bars{margin-top:22px;}
.bar{display:grid;grid-template-columns:190px 1fr 110px;gap:14px;align-items:center;padding:6px 0;color:var(--muted);font-size:0.95rem;}
.bar b{color:var(--text);font-weight:600;text-align:right;font-variant-numeric:tabular-nums;}
.bt{height:10px;background:#1B324A;border-radius:5px;overflow:hidden;}
.bt i{display:block;height:100%;background:#5B7690;border-radius:5px;}
.saved{margin:10px 0 0 0;color:var(--muted);font-size:0.95rem;max-width:64ch;}
.facts{margin:18px 0 0 0;color:var(--muted);}
@media (max-width:640px){.lane{grid-template-columns:110px 1fr;}.lend{display:none;}.bar{grid-template-columns:1fr 100px;}.bar span{grid-column:1 / -1;}.brand h1{font-size:2.3rem;}}
@media (prefers-reduced-motion:reduce){.lane.taken .track i{animation:none;}}
"""


def inject_css(css):
    # strip blank lines and indentation so the markdown parser leaves the CSS alone
    css = "\n".join(line.strip() for line in css.splitlines() if line.strip())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def money(x):
    return f"&#36;{x:.6f}"


def lane_models():
    try:
        with open("config/routing.yaml", "r") as f:
            cfg = yaml.safe_load(f)["routing"]
        return {i: cfg[f"tier_{i}"]["model_key"] for i in (1, 2, 3)}
    except Exception:
        return {}


def route_html(tier, model_key):
    models = lane_models()
    name, color = TIERS.get(tier, ("Unknown", "#8DA2B6"))
    out = [f'<div class="route"><p class="route-note">Your prompt took the <b style="color:{color}">{name}</b> lane and was answered by {html.escape(str(model_key))}.</p>']
    for i in (1, 2, 3):
        lname, lcolor = TIERS[i]
        taken = " taken" if i == tier else ""
        lane_model = html.escape(str(models.get(i, "")))
        end = "Answered here" if i == tier else ""
        out.append(f'<div class="lane{taken}" style="--c:{lcolor}"><div class="lname"><b>{lname}</b><span>{lane_model}</span></div><div class="track"><i></i></div><div class="lend">{end}</div></div>')
    out.append("</div>")
    return "".join(out)


def ledger_html(r, transcription, transcribe_s, media_label):
    color = TIERS.get(r["predicted_tier"], ("Unknown", "#8DA2B6"))[1]
    llm_cost = r["final_cost_usd"]
    stt_cost = transcription.cost_usd if transcription else 0.0
    total = llm_cost + stt_cost
    in_tok = r.get("input_tokens", 0)
    out_tok = r.get("output_tokens", 0)
    ref = in_tok * REF_IN + out_tok * REF_OUT
    total_time = r["latency_ms"] / 1000 + transcribe_s

    out = ['<div class="ledger">']
    out.append(f'<div class="lrow"><span>Model call</span><b>{money(llm_cost)}</b></div>')
    if transcription:
        out.append(f'<div class="lrow"><span>Turning {transcription.duration_s:.1f} seconds of {media_label.lower()} into text</span><b>{money(stt_cost)}</b></div>')
    out.append(f'<div class="lrow total"><span>Total</span><b>{money(total)}</b></div>')
    out.append("</div>")
    if ref > 0:
        pct = max(llm_cost / ref * 100, 1.5)
        saved = (ref - llm_cost) / ref * 100
        out.append('<div class="bars">')
        out.append(f'<div class="bar"><span>GPT-4o would charge</span><div class="bt"><i style="width:100%"></i></div><b>{money(ref)}</b></div>')
        out.append(f'<div class="bar"><span>This model call</span><div class="bt"><i style="width:{pct:.1f}%;background:{color}"></i></div><b>{money(llm_cost)}</b></div>')
        out.append(f'<p class="saved">{saved:.1f}% less on the model call. GPT-4o is a price reference only and is never called.</p>')
        out.append("</div>")
    out.append(f'<p class="facts">{in_tok} tokens in, {out_tok} tokens out, {total_time:.1f} seconds from start to answer.</p>')
    return "".join(out)


def show_result(r, transcription=None, transcribe_s=0.0, media_label="Audio"):
    color = TIERS.get(r["predicted_tier"], ("Unknown", "#8DA2B6"))[1]

    st.markdown(f'<div class="rule" style="--c:{color}"></div>', unsafe_allow_html=True)
    st.markdown("### Answer")
    st.markdown(r["final_response_text"])

    st.markdown("### How it was routed")
    st.markdown(route_html(r["predicted_tier"], r["routed_model_key"]), unsafe_allow_html=True)

    st.markdown("### What it cost")
    st.markdown(ledger_html(r, transcription, transcribe_s, media_label), unsafe_allow_html=True)

    if transcription:
        with st.expander("Transcript"):
            st.write(transcription.text)


inject_css(CSS)

st.markdown(
    '<div class="brand"><span class="mark"><i></i><i></i><i></i></span><h1>Autopilot</h1></div>'
    '<p class="tagline">Send a prompt as text, audio or video. The cheapest model that can handle it answers, and you see what it cost.</p>',
    unsafe_allow_html=True,
)

tab_text, tab_audio, tab_video = st.tabs(["Text", "Audio", "Video"])

with tab_text:
    prompt = st.text_area("Your prompt", height=120, placeholder="e.g. What is the capital of India?")
    if st.button("Get answer", key="run_text"):
        if not prompt.strip():
            st.warning("Type a prompt first.")
        else:
            try:
                with st.spinner("Routing and answering..."):
                    r = process_request(prompt, verify=False)
                show_result(r)
            except Exception as e:
                st.error(str(e))

with tab_audio:
    uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "webm", "mp4"], key="audio_upload")
    recorded = st.audio_input("Or record your prompt") if hasattr(st, "audio_input") else None
    instruction = st.text_input("What should I do with the audio?", "Answer the question in the recording.")

    audio_file = recorded if recorded is not None else uploaded
    if audio_file is not None:
        st.audio(audio_file)

    if st.button("Get answer", key="run_audio"):
        if audio_file is None:
            st.warning("Upload or record audio first.")
        else:
            name = getattr(audio_file, "name", "") or "recording.wav"
            suffix = os.path.splitext(name)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_file.getvalue())
                path = tmp.name
            try:
                t0 = time.perf_counter()
                with st.spinner("Transcribing..."):
                    t = transcribe_audio(path)
                transcribe_s = time.perf_counter() - t0
                with st.spinner("Routing and answering..."):
                    full_prompt = f"{instruction}\n\nTranscript:\n{t.text}"
                    r = process_request(full_prompt, verify=False, route_text=instruction)
                show_result(r, t, transcribe_s, media_label="Audio")
            except Exception as e:
                st.error(str(e))
            finally:
                os.remove(path)

with tab_video:
    st.caption("This version uses the speech in the video. It can't see the picture yet.")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "webm", "mkv", "avi", "m4v"], key="video_upload")
    v_instruction = st.text_input("What should I do with the video?", "Answer the question in the video.", key="video_instruction")

    if video_file is not None:
        st.video(video_file)

    if st.button("Get answer", key="run_video"):
        if video_file is None:
            st.warning("Upload a video first.")
        else:
            suffix = os.path.splitext(video_file.name)[1] or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(video_file.getvalue())
                path = tmp.name
            try:
                t0 = time.perf_counter()
                with st.spinner("Extracting audio and transcribing..."):
                    t = transcribe_video(path)
                transcribe_s = time.perf_counter() - t0
                with st.spinner("Routing and answering..."):
                    full_prompt = f"{v_instruction}\n\nTranscript of the video (if it contains a question, answer it using your own knowledge):\n{t.text}"
                    r = process_request(full_prompt, verify=False, route_text=v_instruction)
                show_result(r, t, transcribe_s, media_label="Video")
            except Exception as e:
                st.error(str(e))
            finally:
                os.remove(path)