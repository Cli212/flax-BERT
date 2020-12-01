from transformers import BertTokenizer, BertForMaskedLM
from bert import FlaxBertForMaskedLM, FlaxBertPredictMask
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
flax_model = FlaxBertForMaskedLM.from_pretrained("bert-base-uncased")
predictions = FlaxBertPredictMask(flax_model,tokenizer)
result = predictions("Paris is the [MASK] of France.")

#
# from transformers import pipeline
# unmaker = pipeline("fill-mask", model="models/bert-base-uncased",tokenizer="bert-base-uncased")
# result = unmaker("Paris is the [MASK] of France.")