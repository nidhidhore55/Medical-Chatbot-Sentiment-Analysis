"""
retriever.py

Retrieves medical answers from:
1. MedQuAD XML dataset
2. Dynamically updated ChromaDB knowledge base
"""

import os
import re
import xml.etree.ElementTree as ET

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import chromadb


class MedQuADRetriever:

    STOP_WORDS = {
        "the", "is", "are", "a", "an", "of", "to", "for",
        "in", "on", "with", "and", "or", "what", "how",
        "can", "i", "my", "do", "does", "about"
    }

    def __init__(
        self,
        dataset_path="Data/MedQuAD",
        database_path="chroma_db"
    ):

        # -------------------------------
        # MedQuAD setup
        # -------------------------------

        self.dataset_path = dataset_path
        self.qa_pairs = []

        self.vectorizer = TfidfVectorizer()
        self.question_vectors = None
        self.questions = []

        # -------------------------------
        # ChromaDB setup
        # -------------------------------

        self.client = chromadb.PersistentClient(
            path=database_path
        )

        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge"
        )

    # ==========================================
    # MEDQUAD
    # ==========================================

    def load_dataset(self):
        """
        Load every XML file inside the MedQuAD dataset.
        """

        for root, dirs, files in os.walk(
            self.dataset_path
        ):

            for file in files:

                if file.endswith(".xml"):

                    filepath = os.path.join(
                        root,
                        file
                    )

                    self._read_xml(filepath)

        # Build TF-IDF index

        self.questions = [
            qa["question"]
            for qa in self.qa_pairs
        ]

        if self.questions:

            self.question_vectors = (
                self.vectorizer.fit_transform(
                    self.questions
                )
            )

        print(
            f"Loaded {len(self.qa_pairs)} "
            "question-answer pairs."
        )

    def _read_xml(self, filepath):
        """
        Read one XML file and extract
        Question & Answer.
        """

        try:

            tree = ET.parse(filepath)

            root = tree.getroot()

            for qa in root.findall(".//QAPair"):

                question = qa.findtext(
                    "Question"
                )

                answer = qa.findtext(
                    "Answer"
                )

                if question and answer:

                    self.qa_pairs.append(
                        {
                            "question": question.strip(),
                            "answer": answer.strip()
                        }
                    )

        except Exception as e:

            print(
                f"Error reading {filepath}: {e}"
            )

    # ==========================================
    # TEXT CLEANING
    # ==========================================

    def clean_text(self, text):
        """
        Clean text by converting to lowercase,
        removing punctuation and stop words.
        """

        words = re.findall(
            r"\b\w+\b",
            text.lower()
        )

        return [
            word
            for word in words
            if word not in self.STOP_WORDS
        ]

    # ==========================================
    # MEDQUAD SEARCH
    # ==========================================

    def search_medquad(self, user_question):
        """
        Search the MedQuAD dataset.
        """

        if (
            self.question_vectors is None
            or not self.qa_pairs
        ):
            return None

        query_vector = (
            self.vectorizer.transform(
                [user_question]
            )
        )

        similarities = cosine_similarity(
            query_vector,
            self.question_vectors
        ).flatten()

        best_index = similarities.argmax()

        best_score = similarities[best_index]

        if best_score > 0:

            return {
                "answer":
                    self.qa_pairs[best_index]["answer"],
                "source":
                    "MedQuAD",
                "score":
                    float(best_score)
            }

        return None

    # ==========================================
    # CHROMADB SEARCH
    # ==========================================

    def search_dynamic_knowledge(
        self,
        user_question
    ):
        """
        Search dynamically updated information
        stored in ChromaDB.
        """

        try:

            count = self.collection.count()

            if count == 0:
                return None

            results = self.collection.query(
                query_texts=[user_question],
                n_results=1
            )

            documents = results.get(
                "documents",
                []
            )

            metadatas = results.get(
                "metadatas",
                []
            )

            if (
                not documents
                or not documents[0]
            ):
                return None

            answer = documents[0][0]

            source = "Dynamic Knowledge Base"

            if (
                metadatas
                and metadatas[0]
                and metadatas[0][0]
            ):

                source = metadatas[0][0].get(
                    "source",
                    source
                )

            return {
                "answer": answer,
                "source": source,
                "score": 1.0
            }

        except Exception as e:

            print(
                f"ChromaDB search error: {e}"
            )

            return None

    # ==========================================
    # COMBINED SEARCH
    # ==========================================

    def search(self, user_question):
        """
        Search both MedQuAD and the dynamic
        ChromaDB knowledge base.
        """

        medquad_result = (
            self.search_medquad(
                user_question
            )
        )

        dynamic_result = (
            self.search_dynamic_knowledge(
                user_question
            )
        )

        # Prefer dynamically updated knowledge
        # when it is available.

        if dynamic_result:

            return dynamic_result

        if medquad_result:

            return medquad_result

        return None

    # ==========================================
    # GET ANSWER
    # ==========================================

    def get_answer(self, user_question):
        """
        Return the answer text.
        """

        result = self.search(
            user_question
        )

        if result:

            return result["answer"]

        return (
            "Sorry, I couldn't find a relevant "
            "answer in the medical knowledge base."
        )


# ==========================================
# MANUAL TEST
# ==========================================

if __name__ == "__main__":

    retriever = MedQuADRetriever()

    retriever.load_dataset()

    print(
        "\nDynamic knowledge entries:",
        retriever.collection.count()
    )

    while True:

        question = input(
            "\nAsk a medical question "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":
            break

        result = retriever.search(
            question
        )

        print("\nAnswer:")

        if result:

            print(result["answer"])

            print(
                f"\nSource: {result['source']}"
            )

        else:

            print(
                "Sorry, I couldn't find a "
                "relevant answer."
            )