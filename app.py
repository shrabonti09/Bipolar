import streamlit as st
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier

@st.cache_resource
def load():
    embedder = SentenceTransformer('l3cube-pune/bengali-sentence-bert-nli')
    model = XGBClassifier()
    model.load_model('bipolar_model.json')  # tomari 78.66% model
    return embedder, model

embedder, model = load()

st.title("BipolarGuard – Bangla + English Mood Detector")
text = st.text_area("Paste any social media post:", height=150)

if st.button("Detect Mood"):
    emb = embedder.encode([text])
    pred = model.predict(emb)[0]
    prob = model.predict_proba(emb)[0]
    labels = ["Neutral", "Manic Episode", "Depressive Episode"]
    st.write(f"**Prediction: {labels[pred]}**")
    st.progress(float(prob[pred]))
    st.write(f"Confidence: {prob[pred]:.1%}")