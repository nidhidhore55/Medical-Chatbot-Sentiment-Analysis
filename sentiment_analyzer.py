"""
sentiment_analyzer.py

Standalone sentiment detection module for the customer-support chatbot.
Drop this file into your existing LangChain project directory.

Design goals (mapped to the internship task):
- Detect positive / negative / neutral sentiment in user messages.
- Be fast enough to run on every turn of a conversation.
- Degrade gracefully if heavy ML dependencies aren't installed.

Primary backend: HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest`
  (a proper 3-class model: negative / neutral / positive - most sentiment
  models are only 2-class positive/negative, which doesn't fit the task spec).
Fallback backend: VADER (nltk.sentiment.vader), a lightweight rule-based
  analyzer that needs no GPU/model download - used automatically if
  `transformers`/`torch` aren't available in your environment.
"""

from dataclasses import dataclass
from typing import Literal

Sentiment = Literal["positive", "negative", "neutral"]


@dataclass
class SentimentResult:
    label: Sentiment
    score: float          # confidence 0-1 for the winning label
    raw_scores: dict       # scores for all three classes, for logging/eval


class SentimentAnalyzer:
    """
    Usage:
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze("This is the third time it's broken, I'm furious.")
        print(result.label, result.score)
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._pipeline = None
        self._vader = None

        if backend in ("auto", "transformer"):
            try:
                self._init_transformer_backend()
                self.backend = "transformer"
                return
            except Exception as e:
                if backend == "transformer":
                    raise
                # fall through to VADER
                self._init_load_warning = str(e)

        self._init_vader_backend()
        self.backend = "vader"

    # ------------------------------------------------------------------ #
    # Backend setup
    # ------------------------------------------------------------------ #
    def _init_transformer_backend(self):
        from transformers import pipeline
        self._pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,  # return all class scores
        )

    def _init_vader_backend(self):
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, text: str) -> SentimentResult:
        if not text or not text.strip():
            return SentimentResult("neutral", 1.0, {"neutral": 1.0, "positive": 0.0, "negative": 0.0})

        if self.backend == "transformer":
            return self._analyze_transformer(text)
        return self._analyze_vader(text)

    # ------------------------------------------------------------------ #
    # Backend implementations
    # ------------------------------------------------------------------ #
    def _analyze_transformer(self, text: str) -> SentimentResult:
        # cardiffnlp labels: 'negative', 'neutral', 'positive'
        scores = self._pipeline(text[:512])[0]  # list of {'label', 'score'}
        raw = {s["label"]: s["score"] for s in scores}
        top_label = max(raw, key=raw.get)
        return SentimentResult(top_label, raw[top_label], raw)

    def _analyze_vader(self, text: str) -> SentimentResult:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        raw = {
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }
        return SentimentResult(label, abs(compound) if compound != 0 else scores["neu"], raw)


if __name__ == "__main__":
    # Quick manual smoke test
    analyzer = SentimentAnalyzer()
    print(f"Backend in use: {analyzer.backend}")
    samples = [
        "Thank you so much, this solved my problem instantly!",
        "This is the third time your product has failed me. Absolutely ridiculous.",
        "Can you tell me your return policy?",
    ]
    for s in samples:
        r = analyzer.analyze(s)
        print(f"[{r.label:8s} | {r.score:.2f}] {s}")