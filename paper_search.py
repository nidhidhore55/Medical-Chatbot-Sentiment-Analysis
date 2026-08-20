from data_loader import ArxivDataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PaperSearcher:

    def __init__(self):
        loader = ArxivDataLoader()
        self.papers = loader.load_papers()

        # Combine title and abstract for searching
        self.documents = [
            f"{paper['title']} {paper['abstract']}"
            for paper in self.papers
        ]

        # Create TF-IDF index
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=50000
        )

        self.paper_vectors = self.vectorizer.fit_transform(
            self.documents
        )

    def search(self, query, top_k=5):
        """Return the most relevant papers for a query."""

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.paper_vectors
        ).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            if similarities[index] > 0:

                paper = self.papers[index].copy()
                paper["similarity"] = float(similarities[index])

                results.append(paper)

        return results


if __name__ == "__main__":

    searcher = PaperSearcher()

    query = input(
        "\nEnter a Computer Science research topic: "
    )

    results = searcher.search(query)

    print("\nRelevant papers:\n")

    for i, paper in enumerate(results, 1):

        print(f"{i}. {paper['title']}")
        print(f"   Categories: {paper['categories']}")
        print(f"   Similarity: {paper['similarity']:.3f}")
        print()