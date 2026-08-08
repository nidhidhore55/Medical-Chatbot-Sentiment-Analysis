import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from retriever import MedQuADRetriever
from entity_recognition import MedicalEntityRecognizer
from sentiment_analyzer import SentimentAnalyzer

st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Q&A Chatbot")
st.write("Ask any medical question based on the MedQuAD dataset.")

# Load dataset only once
@st.cache_resource
def load_chatbot():
    retriever = MedQuADRetriever()
    retriever.load_dataset()

    ner = MedicalEntityRecognizer()
    sentiment_analyzer = SentimentAnalyzer()

    return retriever, ner, sentiment_analyzer


retriever, ner, sentiment_analyzer = load_chatbot()

user_question = st.text_input(
    "Enter your medical question"
)

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        entities = ner.extract_entities(user_question)
        sentiment = sentiment_analyzer.analyze(user_question)
        answer = retriever.get_answer(user_question)

        st.subheader("Sentiment")
        st.write(f"**Detected:** {sentiment.label.capitalize()}")
        st.write(f"**Confidence:** {sentiment.score:.2%}")

        st.subheader("Medical Entities")
        if any(entities.values()):
            for category, values in entities.items():
                if values:
                    st.write(f"**{category}:** {', '.join(values)}")
        else:
            st.write("No medical entities detected.")

        st.subheader("Answer")
        if answer:
            st.success(answer)
        else:
            st.warning("Sorry, I couldn't find a relevant answer in the MedQuAD dataset.")

st.markdown("---")
st.caption("Educational purposes only. This chatbot is not a substitute for professional medical advice.")