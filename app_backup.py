import os
import tempfile
import time

import streamlit as st

from core.audio import transcribe_audio
from core.pipeline import process_request
from core.video import transcribe_video

TIER_NAMES = {1: "Simple", 2: "Moderate", 3: "Complex"}
REF_IN = 2.50 / 1_000_000    # GPT-4o reference price, not called
REF_OUT = 10.00 / 1_000_000

st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("🎯 LLM Cost Autopilot")
st.caption("Give a text, audio or video prompt. The router picks the model and shows the answer, tier and cost.")


def show_result(r, transcription=None, transcribe_s=0.0, media_label="Audio"):
    llm_cost = r["final_cost_usd"]
    stt_cost = transcription.cost_usd if transcription else 0.0
    total_cost = llm_cost + stt_cost
    in_tok = r.get("input_tokens", 0)
    out_tok = r.get("output_tokens", 0)
    ref_cost = in_tok * REF_IN + out_tok * REF_OUT
    total_time = r["latency_ms"] / 1000 + transcribe_s

    st.subheader("Answer")
    st.markdown(r["final_response_text"])

    st.markdown("---")
    st.subheader("Routing and cost")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tier", f"{r['predicted_tier']} ({TIER_NAMES.get(r['predicted_tier'], '?')})")
    c2.metric("Model", r["routed_model_key"])
    c3.metric("LLM cost", f"${llm_cost:.6f}")
    c4.metric("Total cost", f"${total_cost:.6f}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Input tokens", in_tok)
    d2.metric("Output tokens", out_tok)
    d3.metric("Total time", f"{total_time:.1f}s")
    if ref_cost > 0:
        saved = (ref_cost - llm_cost) / ref_cost * 100
        d4.metric("LLM cost saved vs GPT-4o", f"{saved:.1f}%",
                  help="Reference only. GPT-4o is not called; its published price is applied to the same tokens.")

    if transcription:
        st.subheader(media_label)
        a1, a2, a3 = st.columns(3)
        a1.metric(f"{media_label} length", f"{transcription.duration_s:.1f}s")
        a2.metric("Transcription cost", f"${stt_cost:.6f}")
        a3.metric("Transcription time", f"{transcribe_s:.1f}s")
        with st.expander("Transcript"):
            st.write(transcription.text)


tab_text, tab_audio, tab_video = st.tabs(["✍️ Text prompt", "🎙️ Audio prompt", "🎬 Video prompt"])

with tab_text:
    prompt = st.text_area("Your prompt", height=120, placeholder="e.g. What is the capital of India?")
    if st.button("Run", key="run_text"):
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

    if st.button("Run", key="run_audio"):
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
                show_result(r, t, transcribe_s)
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

    if st.button("Run", key="run_video"):
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