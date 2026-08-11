import json
import requests
from bs4 import BeautifulSoup
import hashlib
import chromadb
from datetime import datetime
import time


class KnowledgeBaseUpdater:

    def __init__(self, database_path="chroma_db"):
        # Create persistent ChromaDB database
        self.client = chromadb.PersistentClient(path=database_path)

        # Create or load collection
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge"
        )

    def add_information(self, text, source="unknown"):
        """
        Add new information to the vector database.
        """

        if not text or not text.strip():
            return

        # Create a unique ID from the content
        document_id = hashlib.md5(
            text.encode("utf-8")
        ).hexdigest()

        # Check whether information already exists
        existing = self.collection.get(ids=[document_id])

        if existing["ids"]:
            print("Information already exists.")
            return

        self.collection.add(
            ids=[document_id],
            documents=[text],
            metadatas=[{
                "source": source,
                "updated": datetime.now().isoformat()
            }]
        )

        print("New information added successfully.")

    def get_information(self, query, n_results=3):
        """
        Search the knowledge base for relevant information.
        """

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results

    def fetch_source(self, url):
        """
        Fetch text information from a specified web source.
        """

        print(f"Fetching: {url}")

        try:
            response = requests.get(
                url,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0"
                }
            )

            response.raise_for_status()

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # Remove unnecessary page elements
            for element in soup(
                ["script", "style", "nav", "footer"]
            ):
                element.decompose()

            paragraphs = soup.find_all("p")

            text_parts = []

            for paragraph in paragraphs:
                text = paragraph.get_text(
                    " ",
                    strip=True
                )

                if len(text) > 50:
                    text_parts.append(text)

            # Keep the fetched information manageable
            text = " ".join(text_parts)

            return text[:10000]

        except Exception as e:
            print(f"Error fetching source: {e}")
            return ""

    def update_from_sources(self, config_file="sources.json"):
        """
        Read configured sources and update ChromaDB.
        """

        print("\nChecking configured sources...")

        try:
            with open(
                config_file,
                "r",
                encoding="utf-8"
            ) as file:

                config = json.load(file)

        except Exception as e:
            print(f"Error reading sources.json: {e}")
            return

        sources = config.get("sources", [])

        for source in sources:

            name = source.get(
                "name",
                "Unknown source"
            )

            url = source.get("url")

            if not url:
                continue

            print(f"\nSource: {name}")

            text = self.fetch_source(url)

            if text:

                self.add_information(
                    text,
                    source=name
                )

        print("\nKnowledge base update completed.")


def run_once():
    """
    Run one knowledge-base update.
    """

    updater = KnowledgeBaseUpdater()

    print("Knowledge Base Updater started.")

    updater.update_from_sources()

    # Test retrieval
    results = updater.get_information(
        "How can physical activity help health?"
    )

    print("\nRetrieved information:")

    if results["documents"]:
        for document in results["documents"][0]:
            print(document[:500])
    else:
        print("No information found.")


def periodic_update(interval=3600):
    """
    Periodically update the knowledge base.

    interval is measured in seconds.
    Default: 3600 seconds = 1 hour.
    """

    updater = KnowledgeBaseUpdater()

    while True:

        print("\nChecking for new information...")

        updater.update_from_sources()

        print(
            f"Next update in {interval} seconds."
        )

        time.sleep(interval)


if __name__ == "__main__":
    periodic_update(interval=3600)