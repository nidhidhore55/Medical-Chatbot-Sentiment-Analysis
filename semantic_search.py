import os
import numpy as np
from data_loader import ArxivDataLoader
from sentence_transformers import SentenceTransformer


class SemanticPaperSearcher:

    def __init__(self):

        print("Loading paper dataset...")

        loader = ArxivDataLoader()
        self.papers = loader.load_papers()

        print(f"Loaded {len(self.papers)} papers.")

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.embedding_file = "paper_embeddings.npy"

        if os.path.exists(self.embedding_file):

            print("Loading saved paper embeddings...")

            self.embeddings = np.load(
                self.embedding_file
            )

        else:

            print("Creating paper embeddings...")
            print("This will happen only once.")

            self.documents = [
                f"{paper['title']}. {paper['abstract']}"
                for paper in self.papers
            ]

            self.embeddings = self.model.encode(
                self.documents,
                show_progress_bar=True,
                normalize_embeddings=True
            )

            np.save(
                self.embedding_file,
                self.embeddings
            )

            print("Embeddings saved successfully!")

        print("Semantic search index ready!")

    def search(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []

        for index in top_indices:

            paper = self.papers[index].copy()

            paper["similarity"] = float(
                scores[index]
            )

            results.append(paper)

        return results


if __name__ == "__main__":

    searcher = SemanticPaperSearcher()

    query = input(
        "\nEnter a Computer Science research topic: "
    )

    results = searcher.search(query)

    print("\nMost relevant papers:\n")

    for i, paper in enumerate(results, 1):

        print(f"{i}. {paper['title']}")
        print(f"   Category: {paper['categories']}")
        print(f"   Similarity: {paper['similarity']:.3f}")
        print()