from data_loader import ArxivDataLoader
from semantic_search import SemanticPaperSearcher
from summarizer import PaperSummarizer


class ResearchEngine:

    def __init__(self):

        print("Initializing Research Engine...")

        self.searcher = SemanticPaperSearcher()
        self.summarizer = PaperSummarizer()

        print("Research Engine ready!")

    def search_papers(self, query, top_k=5):

        return self.searcher.search(
            query,
            top_k=top_k
        )

    def summarize_paper(self, paper):

        abstract = paper.get("abstract", "")

        return self.summarizer.summarize(
            abstract
        )


if __name__ == "__main__":

    engine = ResearchEngine()

    query = input(
        "\nEnter a research topic: "
    )

    papers = engine.search_papers(
        query,
        top_k=5
    )

    print("\nRelevant papers:\n")

    for i, paper in enumerate(papers, 1):

        print(f"{i}. {paper['title']}")

    choice = input(
        "\nEnter paper number to summarize: "
    )

    try:

        index = int(choice) - 1

        selected_paper = papers[index]

        print("\nGenerating summary...\n")

        summary = engine.summarize_paper(
            selected_paper
        )

        print("Summary:")
        print(summary)

    except (ValueError, IndexError):

        print("Invalid paper selection.")