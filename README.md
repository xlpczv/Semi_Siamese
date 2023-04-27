# Semi-Siamese Bi-encoder Neural Ranking Model Using Lightweight Fine-Tuning

This github repository is for the paper [*Semi-Siamese Bi-encoder Neural Ranking Model UsingLightweight Fine-Tuning*](https://arxiv.org/pdf/2110.14943) published at WWW 2022. \
We provide codes for *full fine-tuning (FFT)*, *prompt-tuning*, *prefix-tuning*, and *low-rank adaptation (LoRA)*. \

## Package
We used packages listed below.
```
python=3.8.10
torch=1.7.1+cu110
transformers=4.12.5
tensorboard=2.5.0
pytools=2021.2.7
```

## Dataset
We used Robust04b, ClueWeb09b, and MS-MARCO. \
You can download the datasets we used [here](https://drive.google.com/drive/folders/1f8zJ61L7t4DzGnDqNKbHBykwoLADg7Az?usp=sharing). \
In the experiments, we use the name 'robust', 'wt' and 'msmarco' for Robust04, ClueWeb09b, and MS-MARCO respectively. \
Robust04 and ClueWeb09b datasets are both divided into five folds, while MS-MARCO has only one fold. \
When you train the model on MS-MARCO, you should set the argument 'msmarco' as True, and you can limit the train data size by setting the argument 'batches_per_epoch'. \
We used 1024 as 'batches_per_epoch' when training the models using MS-MARCO.

## Model
We adopted BERT (a pre-trained model named 'bert-base-uncased' provided by huggingface) as a backbone model. \
We implemented BERT-based neural ranking models (MonoBERT, ColBERT, and TwinBERT) and trained them. \

## Hybrid LFT
To apply prefix-tuning and LoRA sequentially, you can use files in LoRA folder.
