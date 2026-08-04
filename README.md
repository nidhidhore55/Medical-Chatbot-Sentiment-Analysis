# 🩺 Medical Chatbot with Sentiment Analysis

## 📌 Project Overview

This project is a Medical Chatbot integrated with Sentiment Analysis using Natural Language Processing (NLP).

The chatbot detects the user's emotional sentiment before generating a response.

Supported sentiments:

- 😊 Positive
- 😐 Neutral
- 😔 Negative

The chatbot changes its response tone according to the detected sentiment while providing medical guidance from a simple medical knowledge base.

---

## Features

- Sentiment Detection
- Medical Question Answering
- Emotion-aware Responses
- Confidence Score Display
- Simple Command Line Interface
- Modular Python Design

---

## Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- NLTK

---

## Project Structure

```
Medical-Chatbot-Sentiment-Analysis/

├── app.py
├── chatbot_with_sentiment.py
├── sentiment_analyzer.py
├── evaluate_sentiment.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Chatbot

```bash
python app.py
```

---

## Run Evaluation

```bash
python evaluate_sentiment.py
```

---

## Example

User:

```
I have had a fever for two days.
```

Detected Sentiment:

```
Neutral
```

Bot:

```
Thank you for your question.

A fever is usually caused by an infection...
```

---

## Future Improvements

- Voice Support
- GUI using Streamlit
- Medical Dataset Integration
- Conversation History
- Doctor Recommendation

---

## Author

Created as part of an Internship Project on Sentiment Analysis Integration using Python and NLP.
