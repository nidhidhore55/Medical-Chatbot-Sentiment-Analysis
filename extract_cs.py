import json
from pathlib import Path

# CHANGE THIS to the location of your downloaded arXiv file
SOURCE_FILE = Path(
    r"C:\Users\hp\Downloads\archive\arxiv-metadata-oai-snapshot.json"
)

OUTPUT_FILE = Path("data/arxiv_cs.json")

# Number of Computer Science papers to keep
MAX_PAPERS = 10000

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

count = 0
total = 0

print("Starting arXiv Computer Science extraction...")
print("This may take some time because the source file is about 5 GB.")

with open(SOURCE_FILE, "r", encoding="utf-8") as source, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as output:

    for line in source:
        total += 1

        try:
            paper = json.loads(line)
        except json.JSONDecodeError:
            continue

        categories = paper.get("categories", "")

        # Keep Computer Science papers
        if any(category.startswith("cs.") for category in categories.split()):
            selected = {
                "id": paper.get("id", ""),
                "title": paper.get("title", "").strip(),
                "abstract": paper.get("abstract", "").strip(),
                "authors": paper.get("authors", ""),
                "categories": categories,
                "update_date": paper.get("update_date", "")
            }

            output.write(json.dumps(selected) + "\n")
            count += 1

            if count % 1000 == 0:
                print(f"Collected {count} CS papers...")

            if count >= MAX_PAPERS:
                break

        if total % 100000 == 0:
            print(f"Processed {total:,} records...")

print("\nExtraction completed!")
print(f"Computer Science papers saved: {count}")
print(f"Output file: {OUTPUT_FILE}")