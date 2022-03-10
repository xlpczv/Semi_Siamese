# Prefix-tuning
Follwing [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190), we implement codes for prefix-tuning. \
[Li and Liang](https://arxiv.org/pdf/2101.00190) prepended the prefix to the key and value representations, but we chose to prepend the prefix to the representations right before the self-attention projection.

You can choose 'model' among ['pmonobert', 'p0colbert', 'p2colbert', 'pcolbert', 'p0twinbert', 'p2twinbert', 'ptwinbert']. \
When you choose 'model' starting with 'p0', you can train the same prefixes for both query and document. \
Models starting with 'p2' train Semi-Siamese prefixes, and models starting with only 'p' train different prefixes for query and document.

You can decide from which layer to apply prefixes by setting 'ptune_start'. \
The argument 'hsize' stands for the dimension of hidden states used when generating prefixes, and 'hlen' stands for the length of prefixes.

```
bash run.sh
```