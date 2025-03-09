import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "jempf/optimized-swiss-whisper"  # Replace with your model

class Predictor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model.eval()

    def predict(self, input_text: str):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
