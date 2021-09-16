import torch

import flash
from flash.core.data.utils import download_data
from flash.text import TextClassificationData, TextClassifier

lang='eng' # 'eng', 'kor'


# 1. Create the DataModule
if lang == 'eng':
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
    datamodule = TextClassificationData.from_csv(
        "review",
        "sentiment",
        train_file="data/imdb/train.csv",
        val_file="data/imdb/valid.csv",
        backbone="prajjwal1/bert-medium",
    )
if lang == 'kor':
    datamodule = TextClassificationData.from_csv(
        "review",
        "sentiment",
        train_file="data/nsmc/train.csv",
        val_file="data/nsmc/valid.csv",
        backbone="klue/roberta-base",
    )


# 2. Build the task
if lang == 'eng':
    model = TextClassifier(backbone="prajjwal1/bert-medium", num_classes=datamodule.num_classes)
if lang == 'kor':
    model = TextClassifier(backbone="klue/roberta-base", num_classes=datamodule.num_classes, learning_rate=1e-5)



# 3. Create the trainer and finetune the model
if lang == 'eng':
    trainer = flash.Trainer(max_epochs=3, gpus=1)
    trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
if lang == 'kor':
    trainer = flash.Trainer(max_epochs=30, limit_train_batches=64, limit_val_batches=128, gpus=1)
    trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")



# 4. Classify a few sentences!
if lang == 'eng':
    predictions = model.predict(
        [
            "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
            "The worst movie in the history of cinema.",
            "I come from Bulgaria where it 's almost impossible to have a tornado.",
        ]
    )
if lang == 'kor':
    predictions = model.predict(
        [
            "영화는 제법 흥미로웠다.",
            "이걸 영화라고 만들었나?",
            "배우들의 연기력이 그나마 봐줄만 하다.",
        ]
    )
print(predictions)


# 5. Save the model!
trainer.save_checkpoint("text_classification_model.pt")
