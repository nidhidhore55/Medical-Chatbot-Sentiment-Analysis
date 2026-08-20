import streamlit as st
from research_engine import ResearchEngine
from explainer import ResearchExplainer
from transformers import pipeline
import re

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ArXiv Research Assistant",
    page_icon="🔬",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD MODELS / COMPONENTS
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    engine = ResearchEngine()
    explainer = ResearchExplainer()

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    return engine, explainer, summarizer


engine, explainer, summarizer = load_models()

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🔬 ArXiv Research Assistant")

st.markdown(
    """
    **An AI-powered research chatbot for Computer Science papers**

    Search scientific papers, generate summaries, ask follow-up
    questions, and understand complex research concepts.
    """
)

st.divider()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Settings")

show_visualization = st.sidebar.checkbox(
    "📊 Show Concept Visualization",
    value=True
)

top_k = st.sidebar.slider(
    "📚 Number of papers",
    min_value=1,
    max_value=20,
    value=5
)

# ---------------------------------------------------------
# SEARCH SECTION
# ---------------------------------------------------------

st.header("🔎 Search Research Papers")

query = st.text_input(
    "Enter a research topic",
    placeholder="Example: machine learning for healthcare"
)

search_button = st.button(
    "🔍 Search Papers",
    type="primary"
)

if search_button:

    if not query.strip():
        st.warning("Please enter a research topic.")

    else:

        with st.spinner("Searching research papers..."):

            try:
                results = engine.search(
                    query,
                    top_k=top_k
                )

                st.session_state.results = results
                st.session_state.query = query

            except Exception as e:
                st.error(f"Search error: {e}")


# ---------------------------------------------------------
# DISPLAY SEARCH RESULTS
# ---------------------------------------------------------

if "results" in st.session_state:

    results = st.session_state.results

    st.subheader("📚 Most Relevant Papers")

    if not results:
        st.warning("No relevant papers found.")

    else:

        for i, paper in enumerate(results):

            # Handle dictionary-style results
            if isinstance(paper, dict):

                title = paper.get(
                    "title",
                    "Untitled Paper"
                )

                category = paper.get(
                    "category",
                    paper.get("categories", "Unknown")
                )

                similarity = paper.get(
                    "similarity",
                    paper.get("score", None)
                )

                abstract = paper.get(
                    "abstract",
                    paper.get("summary", "")
                )

            else:
                title = str(paper)
                category = "Unknown"
                similarity = None
                abstract = ""

            with st.expander(
                f"{i + 1}. {title}"
            ):

                st.write(
                    f"**Category:** {category}"
                )

                if similarity is not None:

                    st.write(
                        f"**Similarity:** {similarity:.3f}"
                    )

                if abstract:

                    st.write("### 📄 Abstract")

                    st.write(abstract)

                # -------------------------------------------------
                # SUMMARY BUTTON
                # -------------------------------------------------

                if abstract:

                    if st.button(
                        "📝 Summarize Paper",
                        key=f"summary_{i}"
                    ):

                        with st.spinner(
                            "Generating summary..."
                        ):

                            try:

                                # Limit input length
                                text = abstract[:4000]

                                summary_result = summarizer(
                                    text,
                                    max_length=150,
                                    min_length=40,
                                    do_sample=False
                                )

                                summary = summary_result[0][
                                    "summary_text"
                                ]

                                st.success("Summary")

                                st.write(summary)

                            except Exception as e:

                                st.error(
                                    f"Summarization error: {e}"
                                )


# ---------------------------------------------------------
# AI EXPLANATION SECTION
# ---------------------------------------------------------

st.divider()

st.header("🧠 Ask the Research Assistant")

topic = st.text_input(
    "Enter a concept or research topic to explain",
    placeholder="Example: Explain transformer architecture"
)

if st.button(
    "💡 Explain Concept",
    type="secondary"
):

    if not topic.strip():

        st.warning(
            "Please enter a topic or concept."
        )

    else:

        with st.spinner(
            "Generating explanation..."
        ):

            try:

                explanation = explainer.explain(
                    topic
                )

                st.subheader("📖 Explanation")

                st.write(explanation)

            except Exception as e:

                st.error(
                    f"Explanation error: {e}"
                )


# ---------------------------------------------------------
# FOLLOW-UP QUESTIONS
# ---------------------------------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


st.divider()

st.header("💬 Follow-up Questions")

question = st.text_input(
    "Ask a follow-up question",
    placeholder="Example: Why is this approach useful?"
)

if st.button("Ask Question"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Thinking..."
        ):

            try:

                # Build context from previous conversation
                context = ""

                for item in st.session_state.chat_history:

                    context += (
                        f"User: {item['question']}\n"
                        f"Assistant: {item['answer']}\n"
                    )

                prompt = f"""
You are an expert Computer Science research assistant.

Previous conversation:
{context}

User's question:
{question}

Give a clear and technically accurate answer.
Explain difficult concepts in simple language when appropriate.
"""

                answer = explainer.explain(
                    prompt
                )

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": answer
                    }
                )

                st.subheader("🤖 Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error generating answer: {e}"
                )


# ---------------------------------------------------------
# CONVERSATION HISTORY
# ---------------------------------------------------------

if st.session_state.chat_history:

    st.divider()

    st.subheader("🗨️ Conversation History")

    for item in st.session_state.chat_history:

        st.markdown(
            f"**You:** {item['question']}"
        )

        st.markdown(
            f"**Assistant:** {item['answer']}"
        )


# ---------------------------------------------------------
# CONCEPT VISUALIZATION
# ---------------------------------------------------------

if show_visualization:

    st.divider()

    st.header("📊 Concept Visualization")

    if "results" in st.session_state:

        results = st.session_state.results

        # Collect categories
        categories = []

        for paper in results:

            if isinstance(paper, dict):

                category = paper.get(
                    "category",
                    paper.get("categories", "")
                )

                if category:

                    categories.append(
                        str(category)
                    )

        if categories:

            st.write(
                "### Research Topic Distribution"
            )

            from collections import Counter

            category_counts = Counter(
                categories
            )

            st.bar_chart(
                category_counts
            )

        else:

            st.info(
                "Search for papers to generate a visualization."
            )

    else:

        st.info(
            "Search for a topic first to visualize the research results."
        )


# ---------------------------------------------------------
# INFORMATION EXTRACTION
# ---------------------------------------------------------

if "results" in st.session_state:

    st.divider()

    st.header("🔍 Information Extraction")

    results = st.session_state.results

    all_text = ""

    for paper in results:

        if isinstance(paper, dict):

            abstract = paper.get(
                "abstract",
                paper.get("summary", "")
            )

            all_text += " " + str(abstract)

    if all_text:

        # Extract important technical terms
        words = re.findall(
            r"\b[A-Za-z][A-Za-z-]{3,}\b",
            all_text
        )

        # Remove common words
        stop_words = {
            "this",
            "that",
            "with",
            "from",
            "using",
            "which",
            "these",
            "their",
            "have",
            "been",
            "into",
            "such",
            "than",
            "also",
            "were",
            "where",
            "about",
            "based",
            "more",
            "other"
        }

        important_words = [
            word.lower()
            for word in words
            if word.lower() not in stop_words
        ]

        from collections import Counter

        word_counts = Counter(
            important_words
        )

        st.write(
            "### Frequently Mentioned Terms"
        )

        st.write(
            ", ".join(
                word
                for word, count
                in word_counts.most_common(20)
            )
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "🔬 ArXiv Research Assistant | "
    "Computer Science Research Chatbot"
)