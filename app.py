import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭", layout="centered")

st.title("🎭 Sentiment Analyzer")
st.markdown("Type any text and find out if it's **Positive**, **Negative**, or **Neutral**.")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_model()

st.markdown("### Try an example")
examples = [
    "I absolutely love this product, it changed my life!",
    "This is the worst experience I have ever had.",
    "The weather today is okay, nothing special.",
]
for ex in examples:
    if st.button(ex):
        st.session_state.user_input = ex

user_input = st.text_area(
    "Or type your own text here:",
    value=st.session_state.get("user_input", ""),
    height=150,
    placeholder="Type something..."
)

if st.button("Analyze", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(user_input[:512])[0]
            label = result["label"]
            score = round(result["score"] * 100, 2)

        if label == "POSITIVE":
            st.success(f"Positive — {score}% confidence")
        elif label == "NEGATIVE":
            st.error(f"Negative — {score}% confidence")
        else:
            st.info(f"Neutral — {score}% confidence")

        st.markdown("---")
        st.markdown(f"**Raw output:** Label: `{label}` | Confidence: `{score}%`")

st.markdown("---")
st.caption("Built with Streamlit & HuggingFace Transformers · Ashitha P Sujith")