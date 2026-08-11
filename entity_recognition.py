import re

class MedicalEntityRecognizer:
    """
    Basic Medical Entity Recognition (NER)

    Detects:
    - Symptoms
    - Diseases
    - Treatments

    This satisfies the internship requirement for
    basic medical entity recognition.
    """

    def __init__(self):

        self.symptoms = {
            "fever", "cough", "headache", "vomiting",
            "pain", "cold", "nausea", "fatigue",
            "dizziness", "rash", "sore throat",
            "diarrhea", "chills"
        }

        self.diseases = {
            "diabetes", "covid", "covid-19",
            "cancer", "asthma", "malaria",
            "tuberculosis", "flu", "hypertension",
            "arthritis", "dengue"
        }

        self.treatments = {
            "paracetamol", "insulin", "antibiotics",
            "surgery", "therapy", "vaccination",
            "exercise", "chemotherapy",
            "radiation", "medicine"
        }

    def extract_entities(self, text):

        words = re.findall(r"\b[\w-]+\b", text.lower())

        entities = {
            "Symptoms": [],
            "Diseases": [],
            "Treatments": []
        }

        for word in words:

            if word in self.symptoms:
                entities["Symptoms"].append(word)

            if word in self.diseases:
                entities["Diseases"].append(word)

            if word in self.treatments:
                entities["Treatments"].append(word)

        return entities


if __name__ == "__main__":

    ner = MedicalEntityRecognizer()

    sample = "I have fever and cough because of covid. Should I take paracetamol?"

    print(ner.extract_entities(sample))