"""
chatbot_with_sentiment.py

Shows how to wire `sentiment_analyzer.py` into a LangChain-based chatbot so
the bot recognizes and responds appropriately to positive, negative, or
neutral sentiment in user messages (the internship task's Expected Outcome).

This is written to be adapted, not run as-is: replace `build_base_chain()`
with however your training project already builds its LangChain chain
(e.g. RetrievalQA over your MedQuAD vectorstore, ConversationChain, etc.).
The sentiment step wraps around whatever chain you already have - it does
not replace your existing retrieval/LLM logic.
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from sentiment_analyzer import SentimentAnalyzer, SentimentResult

# Tone instructions injected into the system prompt depending on detected
# sentiment. This is the "respond appropriately to customer emotions" part
# of the task - the retrieval/answer content is unchanged, only the framing.
SENTIMENT_GUIDANCE = {
    "negative": (
        "The user seems frustrated or upset. Open with a brief, genuine "
        "acknowledgement of their frustration before answering. Keep the "
        "tone calm, apologetic where appropriate, and solution-focused. "
        "Do not be dismissive or overly cheerful."
    ),
    "positive": (
        "The user seems happy or satisfied. Match their positive energy "
        "briefly, then continue being helpful and clear."
    ),
    "neutral": (
        "The user's tone is neutral. Respond in a clear, friendly, "
        "professional manner without over- or under-reacting."
    ),
}


def build_base_chain(llm, retriever=None):
    """
    Placeholder for your existing training-project chain.
    Replace this with the actual chain you built during training
    (e.g. a RetrievalQA chain over the MedQuAD dataset).
    """
    system_template = (
        "You are a helpful customer support assistant.\n"
        "{sentiment_guidance}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )
    prompt = ChatPromptTemplate.from_template(system_template)
    return prompt | llm


class SentimentAwareChatbot:
    """
    Wraps an existing LangChain chain with a sentiment-detection step.

    Usage:
        bot = SentimentAwareChatbot(llm=your_llm, retriever=your_retriever)
        answer, meta = bot.respond("I've asked THREE times and nothing works!")
    """

    def __init__(self, llm, retriever=None):
        self.analyzer = SentimentAnalyzer()
        self.chain = build_base_chain(llm, retriever)
        self.retriever = retriever

    def _get_context(self, question: str) -> str:
        if self.retriever is None:
            return ""
        docs = self.retriever.get_relevant_documents(question)
        return "\n\n".join(d.page_content for d in docs)

    def respond(self, user_message: str):
        sentiment: SentimentResult = self.analyzer.analyze(user_message)
        guidance = SENTIMENT_GUIDANCE[sentiment.label]
        context = self._get_context(user_message)

        answer = self.chain.invoke({
            "sentiment_guidance": guidance,
            "context": context,
            "question": user_message,
        })

        meta = {
            "sentiment": sentiment.label,
            "confidence": round(sentiment.score, 3),
            "raw_scores": sentiment.raw_scores,
        }
        return answer, meta


if __name__ == "__main__":
    print(
        "This module is meant to be imported into your existing chatbot "
        "project. Replace build_base_chain() with your training project's "
        "actual chain, then instantiate SentimentAwareChatbot(llm, retriever)."
    )
    # -------------------------------------------------------
    # Generate response according to sentiment
    # -------------------------------------------------------

    def generate_response(self, message, sentiment):

        topic = self.detect_topic(message)

        if topic:
            answer = self.medical_knowledge[topic]
        else:
            answer = (
                "I'm sorry, I don't have enough medical information about that topic. "
                "Please consult a qualified healthcare professional."
            )

        # Modify tone based on sentiment

        if sentiment == "negative":

            prefix = (
                "I'm really sorry that you're feeling this way. "
                "I understand your concern.\n\n"
            )

        elif sentiment == "positive":

            prefix = (
                "I'm glad you're reaching out with a positive attitude!\n\n"
            )

        else:

            prefix = (
                "Thank you for your question.\n\n"
            )

        return prefix + answer

    # -------------------------------------------------------
    # Main chatbot function
    # -------------------------------------------------------

    def respond(self, message):

        result = self.analyzer.analyze(message)

        reply = self.generate_response(
            message,
            result.label
        )

        metadata = {

            "sentiment": result.label,
            "confidence": round(result.score, 3),
            "raw_scores": result.raw_scores

        }

        return reply, metadata
    # -------------------------------------------------------
# Run chatbot from terminal
# -------------------------------------------------------

def main():

    print("=" * 60)
    print("      Medical Chatbot with Sentiment Analysis")
    print("=" * 60)
    print("\nType 'exit' anytime to quit.\n")

    chatbot = SentimentAwareChatbot()

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("\nBot: Thank you for using the Medical Chatbot.")
            print("Stay healthy and take care!")
            break

        response, meta = chatbot.respond(user_input)

        print("\n------------------------------")
        print("Detected Sentiment :", meta["sentiment"].upper())
        print("Confidence         :", meta["confidence"])
        print("------------------------------")
        print("Bot:", response)
        print()


if __name__ == "__main__":
    main()
