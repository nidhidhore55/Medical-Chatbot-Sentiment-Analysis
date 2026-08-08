# Medical Q&A Chatbot using MedQuAD

A Python-based medical question-answering chatbot built using the MedQuAD dataset. The system retrieves relevant medical answers using TF-IDF and cosine similarity and provides basic medical entity recognition and sentiment analysis through a Streamlit interface.

## Features

- Medical question-answering using the MedQuAD dataset
- TF-IDF based information retrieval
- Cosine similarity for finding relevant questions
- Basic medical entity recognition
- Sentiment analysis of user questions
- Interactive Streamlit user interface
- Medical safety disclaimer

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLTK
- XML
- MedQuAD Dataset

## Project Structure

```text
Medical-QA-Chatbot/
│
├── app.py
├── retriever.py
├── entity_recognition.py
├── sentiment_analyzer.py
├── requirements.txt
├── README.md
│
└── Data/
    └── MedQuAD/
        ├── 1_CancerGov_QA/
        ├── 2_GARD_QA/
        ├── 3_GHR_QA/
        ├── ...
        └── 12_MPlusHerbsSupplements_QA/