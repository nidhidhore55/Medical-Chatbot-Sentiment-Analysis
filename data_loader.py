import json
from pathlib import Path


class ArxivDataLoader:
    def __init__(self, data_path="data/arxiv_cs.json"):
        self.data_path = Path(data_path)

    def load_papers(self):
        """Load Computer Science papers from the JSONL dataset."""

        papers = []

        with open(self.data_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    paper = json.loads(line)

                    if paper.get("title") and paper.get("abstract"):
                        papers.append(paper)

                except json.JSONDecodeError:
                    continue

        return papers


if __name__ == "__main__":
    loader = ArxivDataLoader()

    papers = loader.load_papers()

    print(f"Loaded {len(papers)} papers.")

    if papers:
        print("\nFirst paper:")
        print("Title:", papers[0]["title"])
        print("Categories:", papers[0]["categories"])