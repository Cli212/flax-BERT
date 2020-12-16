from transformers import BertTokenizer, BertForMaskedLM
import bert
from bert import FlaxBertForPretrained, FlaxBertPredictMask
import sys
from utils import BertConfig
print(sys.path)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
flax_model = FlaxBertForPretrained.from_pretrained("../models/bert-base-uncased")
predictions = FlaxBertPredictMask(flax_model,tokenizer)
result = predictions("Paris is the [MASK] of France.")

from transformers import FlaxBertModel

model = FlaxBertModel.from_pretrained("bert-base-uncased")
model = FlaxBertModel.from_pretrained("../models/bert-base-uncased/")

#
# from transformers import pipeline
# unmaker = pipeline("fill-mask", model="models/bert-base-uncased",tokenizer="bert-base-uncased")
# result = unmaker("Paris is the [MASK] of France.")

state = flax_model.state1
params = flax_model.state2