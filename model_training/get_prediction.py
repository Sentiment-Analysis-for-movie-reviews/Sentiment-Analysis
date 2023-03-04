import torch


class Get_prediction:
    def __init__(self, review_text: str, tokenizer, device):
        self.review_text = review_text
        self.tokenizer = tokenizer
        self.device = device


    def get_label(self, model):
        encoded_review = self.tokenizer.encode_plus(
            self.review_text,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        # class_names = ["happy","not-relevant","angry","surprise","sad", "disgust"]

        output = model(input_ids, attention_mask)

        _, prediction = torch.max(output[0], dim=1)
        return prediction