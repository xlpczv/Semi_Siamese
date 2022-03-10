#! /bin/bash
data=robust
model_card=bert-base-uncased
model=twinbert

random_seed=41
gpunum=0
MAX_EPOCH=30
freeze_bert=2

# scheduler=linear
# warmup_epoch=1
# weight_decay=0.01

lr=0.0001

lora_attn_dim=16
lora_attn_alpha=32
lora_dropout=0.1

for fold in f1 f2 f3 f4 f5
# for fold in full
do
    if [ ! -z "$1" ]; then
        fold=$1
    fi

    if [ ! -z "$2" ]; then
        random_seed=$2
    fi
    exp="Ldi"$lora_attn_dim"La"$lora_attn_alpha"Ldr"$lora_dropout"init""$random_seed""lr""$lr"

    # # 1. make ./models/$model/weights.p (weights file) in ./models.
    echo "training"
    outdir="$data"_"$model"_"$fold""$exp"
    echo $outdir

    echo $model
    python train.py \
        --model $model \
        --datafiles ../data/$data/queries.tsv ../data/$data/documents.tsv \
        --qrels ../data/$data/qrels \
        --train_pairs ../data/$data/$fold.train.pairs \
        --valid_run ../data/$data/$fold.valid.run \
        --model_out_dir models/$outdir \
        --max_epoch $MAX_EPOCH \
        --gpunum $gpunum \
        --random_seed $random_seed  \
        --freeze_bert $freeze_bert \
        --lr $lr  \
        --lora_attn_dim $lora_attn_dim \
        --lora_attn_alpha $lora_attn_alpha \
        --lora_dropout $lora_dropout \
        # --scheduler $scheduler \
        # --warmup_epoch $warmup_epoch \
        # --weight_decay $weight_decay \
        # --msmarco True \
        # --batches_per_epoch 1024 \

    # 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
    echo "testing"
    python rerank.py \
        --model $model \
        --datafiles ../data/$data/queries.tsv ../data/$data/documents.tsv \
        --run ../data/$data/$fold.test.run \
        --model_weights models/$outdir/weights.p \
        --out_path models/$outdir/test.run \
        --gpunum $gpunum \
        --lora_attn_dim $lora_attn_dim \
        --lora_attn_alpha $lora_attn_alpha \
        --lora_dropout $lora_dropout \

    #3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
    echo "evaluating"
    ../bin/trec_eval -m all_trec ../data/$data/qrels models/$outdir/test.run > models/$outdir/eval.result
done
