# lightning-flash-examples

reference code for lightning-flash

# text-classification

### how to

- installation
```
$ pip install 'lightning-flash[text,serve]'
```

- data preparation
```
# get train.txt, valid.txt from https://github.com/dsindex/iclassifier/tree/master/data/clova_sentiments
$ cd data
$ python convert-to-csv.py --input_path train.txt > train.csv
$ python convert-to-csv.py --input_path valid.txt > valid.csv
```

- train
```
$ python text-classification.py

# 1.8.1+cu101 -> OK
# 1.9.0+cu102 -> Segmentation falut
```

- inference
```
# command line cli
$ flash text_classification

# 1.9.0+cu102 -> Segmentation fault (core dumped)
```

- server
```
$ python server.py

# 1.9.0+cu102 -> Segmentation fault (core dumped)

$ python request.py

```

# references

- https://github.com/PyTorchLightning/lightning-flash
- https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.text.classification.model.TextClassifier.html#flash.text.classification.model.TextClassifier
- https://lightning-flash.readthedocs.io/en/latest/reference/text_classification.html#text-classification
- https://github.com/PyTorchLightning/lightning-flash/blob/master/flash/core/serve/server.py
