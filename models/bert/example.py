import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
input_ids = tokenizer.encode("i {} you".format(tokenizer.mask_token), return_tensors="pt")
outputs = model(input_ids)
prediction_scores = outputs[0]
print(prediction_scores)
predicted_ids = torch.max(prediction_scores, 2)[1].squeeze().tolist()
predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
print(predicted_tokens)
# ['.', 'i', 'love', 'you', '.']
