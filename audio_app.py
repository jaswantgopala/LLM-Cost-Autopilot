import os
import tempfile
import streamlit as st

from core.audio import transcribe_audio
from core.pipeline import process_request

st.title("Audio to answer")

uploaded = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "webm", "mp4"])
instruction = st.text_input("What should I do with it?", "Summarize this recording in 3 bullet points.")

if uploaded is not None:
    st.audio(uploaded)
    if st.button("Run"):
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name
        try:
            with st.spinner("Transcribing..."):
                t = transcribe_audio(path)
            with st.spinner("Routing and answering..."):
                prompt = f"{instruction}\n\nTranscript:\n{t.text}"
                r = process_request(prompt, verify=False, route_text=instruction)

            st.subheader("Answer")
            st.write(r["final_response_text"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Tier", r["predicted_tier"])
            col2.metric("Model", r["routed_model_key"])
            col3.metric("Total cost", f"${t.cost_usd + r['final_cost_usd']:.6f}")

            with st.expander("Transcript"):
                st.write(t.text)
        except Exception as e:
            st.error(str(e))
        finally:
            os.remove(path)