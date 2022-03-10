# Prompt-tuning
Following [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691), we implement codes for prompt-tuning.

You can choose the 'model' among ['pmonobert', 'p0colbert', and 'p0twinbert'] to train MonoBERT, ColBERT, and TwinBERT using prompt-tuning repsectively.

The argument 'hsize' stands for the dimension of hidden states used when generating prompts, and 'hlen' stands for the length of prompts. \

```
bash run.sh
```