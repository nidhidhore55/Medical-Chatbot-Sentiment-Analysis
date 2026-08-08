"""
retriever.py

Loads the MedQuAD dataset from XML files and extracts
medical questions and answers.

This module is responsible for:
1. Reading all XML files
2. Extracting question-answer pairs
3. Returning them as Python dictionaries
"""

import os
import re
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MedQuADRetriever:  
    STOP_WORDS = {
        "the", "is", "are", "a", "an", "of", "to", "for",
        "in", "on", "with", "and", "or", "what", "how",
        "can", "i", "my", "do", "does", "about"
    }
    def __init__(self, dataset_path="data/MedQuAD"):
        self.dataset_path = dataset_path
        self.qa_pairs = []
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = None
        self.questions = []

    def load_dataset(self):
        """
        Load every XML file inside the MedQuAD dataset.
        """

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(".xml"):
                    filepath = os.path.join(root, file)
                    self._read_xml(filepath)
        # Build TF-IDF index
        self.questions = [qa["question"] for qa in self.qa_pairs]

        if self.questions:
            self.question_vectors = self.vectorizer.fit_transform(self.questions)
        print(f"Loaded {len(self.qa_pairs)} question-answer pairs.")

    def _read_xml(self, filepath):
        """
        Read one XML file and extract Question & Answer.
        """

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            for qa in root.findall(".//QAPair"):

                question = qa.findtext("Question")
                answer = qa.findtext("Answer")

                if question and answer:

                    self.qa_pairs.append(
                        {
                            "question": question.strip(),
                            "answer": answer.strip(),
                        }
                    )

        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    def clean_text(self, text):
        """
        Clean text by converting to lowercase, removing punctuation,
        and removing common stop words.
        """

        words = re.findall(r"\b\w+\b", text.lower())

        return {
            word
            for word in words
            if word not in self.STOP_WORDS
        }
        return self.qa_pairs[best_index]

    def get_answer(self, user_question):
        """
        Return only the answer text.
        """

        result = self.search(user_question)

        if result:
            return result["answer"]

        return (
            "Sorry, I couldn't find a relevant answer in the medical knowledge base."
        )if __name__ == "__main__":

    retriever = MedQuADRetriever()

    retriever.load_dataset()

    while True:

        question = input("\nAsk a medical question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        answer = retriever.get_answer(question)

        print("\nAnswer:")
        print(answer)
    