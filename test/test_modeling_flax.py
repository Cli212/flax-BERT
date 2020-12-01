from bert import FlaxBertForMaskedLM
from transformers import BertForMaskedLM, BertTokenizer, TensorType, FlaxBertModel, BertModel
import torch
import unittest
import numpy as np
class FlaxBertModelTest(unittest.TestCase):
    def test_from_pytorch(self):
        with torch.no_grad():
            with self.subTest("bert-base-uncased"):
                tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
                flax_model = FlaxBertForMaskedLM.from_pretrained("bert-base-uncased")
                torch_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
                test_str = "This is a simple test"
                flax_inputs = tokenizer.encode_plus(test_str,return_tensors=TensorType.JAX)
                torch_inputs = tokenizer.encode_plus(test_str,return_tensors=TensorType.PYTORCH)
                flax_outputs = flax_model(**flax_inputs)
                torch_outputs = torch_model(**torch_inputs)[0].numpy()

                self.assertEqual(len(flax_outputs),len(torch_outputs))
                for flax_output,torch_output in zip(flax_outputs,torch_outputs):
                    self.assert_almost_equals(flax_output,torch_output,5e-4)
    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = (a - b).sum()
        self.assertLessEqual(diff, tol, "Difference between torch and flax is {} (>= {})".format(diff, tol))

if __name__ == "__main__":
    unittest.main()