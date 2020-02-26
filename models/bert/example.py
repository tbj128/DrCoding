import torch
from transformers import BertTokenizer, BertForMaskedLM

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
input_ids = tokenizer.encode("I {} you".format(tokenizer.mask_token), return_tensors="tf")
outputs = model(input_ids)
prediction_scores = outputs[0]
print(prediction_scores)
# predicted_ids = tf.reshape(tf.argmax(prediction_scores, -1), [-1])
# predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids)
# print(predicted_tokens)
# ['.', 'i', 'love', 'you', '.']
