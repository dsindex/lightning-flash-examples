# lightning-flash-examples

reference code for lightning-flash

# text-classification

### how to

- installation
```
$ pip install 'lightning-flash[text]'
$ pip install lightning-flash
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
```

- inference
```
# command line cli
$ flash text_classification
```

- server
```
$ python server.py

$ python request.py
```

# references

- https://github.com/PyTorchLightning/lightning-flash
- https://lightning-flash.readthedocs.io/en/latest/api/generated/flash.text.classification.model.TextClassifier.html#flash.text.classification.model.TextClassifier
- https://lightning-flash.readthedocs.io/en/latest/reference/text_classification.html#text-classification

