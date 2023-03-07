import torch

"""This script allows to get label prediction"""
class GetPrediction:
    def __init__(self, input_text: str, tokenizer, device):
        self.input_text = input_text
        self.tokenizer = tokenizer
        self.device = device

    def get_label(self, model):
        encoded_text = self.tokenizer.encode_plus(
            self.input_text,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)

        output = model(input_ids, attention_mask)

        _, prediction = torch.max(output[0], dim=1)
        class_names = ["joy","sadness","surprise","disgust","anger", "fear", "trust", "anticipation"]

        return class_names[prediction]