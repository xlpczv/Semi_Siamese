# LoRA
Low-Rank Adaptation (LoRA) was introduced in [Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685). \
We implement LoRA for neural ranking models.

You can choose 'model' among ['vbert', 'colbert', 'twinbert', 'hqcolbert', 'hvcolbert', 'hqtwinbert', 'hvtwinbert']. \
Models starting with 'hv' are Semi-Siamese models that use different LoRA value weights as in our paper. \
We also provide models using different LoRA query weights as well, and you can implement it by using model names starting with 'hq'. \
Model names without 'hq' or 'hv' are used for Siamese LoRA weights for query and document.

Arguments 'lora_attn_dim', 'lora_attn_alpha', and 'lora_dropout' each stands for rank, alpha, and dropout rate for LoRA described in the paper. \
If you set 'lora_d_attn_dim', 'lora_d_attn_alpha', and 'lora_d_dropout' in addition to 'lora_attn_dim', 'lora_attn_alpha', and 'lora_dropout', you can train LoRA+ that is proposed in our paper.

```
bash run.sh
```

# Hybrid models (prefix-tuning & LoRA)
If you want to train hybrid models using both prefix-tuning and LoRA, you can run either run_hybrid_PL.sh or run_hybrid_LP.sh. \
By setting 'initial_bert_weights' as the file name for weights of a previously-trained model and using 'freeze_prefix' or 'freeze_LoRA', you can sequentially train the models.

```
bash run_hybrid_PL.sh ## Prefix-tuning -> LoRA
bash run_hybrid_LP.sh ## LoRA -> Prefix-tuning
```