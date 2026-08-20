from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PaperSummarizer:

    def __init__(self):

        print("Loading summarization model...")

        model_name = model_name = "sshleifer/distilbart-cnn-6-6"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        print("Summarization model loaded!")

    def summarize(self, text):

        if not text or not text.strip():
            return "No abstract available."

        # Limit input length
        text = text[:4000]

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=50,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary


if __name__ == "__main__":

    summarizer = PaperSummarizer()

    text = input(
        "\nPaste a research paper abstract:\n"
    )

    summary = summarizer.summarize(text)

    print("\nSummary:")
    print(summary)