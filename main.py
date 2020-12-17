from transformers import BertTokenizer, BertForMaskedLM, TensorType, BertModel
import bert
from bert import FlaxBertForPretrained, FlaxBertPredictMask
import sys
from utils import BertConfig
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
flax_model = FlaxBertForPretrained.from_pretrained("../models/bert-base-uncased")
test_flax = FlaxBertPredictMask(flax_model,tokenizer)
result = test_flax("Paris is the [MASK] of France.")

# from transformers import FlaxBertModel
#
# model = FlaxBertModel.from_pretrained("bert-base-uncased")
# model = FlaxBertModel.from_pretrained("../models/bert-base-uncased/")

#
# from transformers import pipeline
# unmaker = pipeline("fill-mask", model="models/bert-base-uncased",tokenizer="bert-base-uncased")
# result = unmaker("Paris is the [MASK] of France.")

