from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ResearchExplainer:

    def __init__(self):

        print("Loading explanation model...")

        model_name = "google/flan-t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        print("Explanation model loaded!")

    def explain(self, question, context=""):

        prompt = f"""
You are a helpful computer science research assistant.

Use the research context below to answer the user's question.

Research context:
{context}

User question:
{question}

Give a clear and simple explanation.
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        output = self.model.generate(
            inputs["input_ids"],
            max_length=200,
            num_beams=4,
            early_stopping=True
        )

        answer = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return answer


if __name__ == "__main__":

    explainer = ResearchExplainer()

    question = input(
        "\nAsk a research question: "
    )

    answer = explainer.explain(question)

    print("\nExplanation:")
    print(answer)