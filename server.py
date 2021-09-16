import torch

from flash.text import TextClassifier

model = TextClassifier.load_from_checkpoint("text_classification_model.pt")
model.serve(host='127.0.0.1', port=8000)
