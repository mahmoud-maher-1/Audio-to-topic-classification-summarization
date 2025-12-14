import streamlit as st
import whisper
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from src.data_preprocessing import preprocess_text
import keras
import gc

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

@st.cache_resource
def load_resources():
    whisper_model = whisper.load_model("base")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model_dist = keras.models.load_model('Models/distributed_types_20classes.keras')
    model_gen = keras.models.load_model('Models/generalized_types_6classes.keras')
    le_dist = joblib.load('Models/label_encoder_dist.joblib')
    le_gen = joblib.load('Models/label_encoder_gen.joblib')
    summarizer_abs = pipeline("summarization", model="facebook/bart-large-cnn")
    return whisper_model, embed_model, model_dist, model_gen, le_dist, le_gen, summarizer_abs


whisper_model, embed_model, model_dist, model_gen, le_dist, le_gen, summarizer_abs = load_resources()


def extractive_summary(text, sentences_count=8):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])


st.title("Audio to Article, Topic Classification & Summarization System")
audio_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format='audio/mp3')

    if st.button("Analyze Audio"):
        raw_text = None

        # --- Step 1: Transcription ---
        with st.spinner("Transcribing..."):
            try:
                import os
                import tempfile

                # Create a temp file to handle the upload safely
                suffix = os.path.splitext(audio_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(audio_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Transcribe
                result = whisper_model.transcribe(tmp_file_path)
                raw_text = result['text']

                st.subheader("Transcribed Text")
                st.write(raw_text)

                # Cleanup temp file
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"Transcription failed: {e}")

        # --- Step 2 & 3: Classify and Summarize (Only if Step 1 succeeded) ---
        if raw_text:
            del whisper_model
            gc.collect()
            with st.spinner("Classifying..."):
                processed_text = preprocess_text(raw_text)
                embedding = np.array(embed_model.encode([processed_text]))

                pred_dist = model_dist.predict(embedding)
                pred_gen = model_gen.predict(embedding)

                label_dist = le_dist.inverse_transform([np.argmax(pred_dist)])[0]
                label_gen = le_gen.inverse_transform([np.argmax(pred_gen)])[0]

                st.subheader("Predictions")
                st.info(f"Specific: {label_dist}")
                st.success(f"General: {label_gen}")

            with st.spinner("Summarizing..."):
                st.markdown("### Extractive")
                st.write(extractive_summary(raw_text))

                st.markdown("### Abstractive")
                try:
                    input_text = raw_text[:3000]
                    summary = summarizer_abs(input_text, max_length=130, min_length=30, do_sample=False)
                    st.write(summary[0]['summary_text'])
                except Exception as e:
                    st.error(f"Abstractive summary failed: {e}")