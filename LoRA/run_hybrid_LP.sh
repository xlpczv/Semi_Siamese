#! /bin/bash
data=msmarco
model_card=bert-base-uncased
lora_model=hvcolbert
model=colbert

random_seed=41
gpunum=0
MAX_EPOCH=3

freeze_bert=2
freeze_lora=True

# scheduler=linear
# warmup_epoch=1
# weight_decay=0

lr=0.0001
p_lr=0.0001

p_start=0
hsize=256
hlen=10

lora_attn_dim=16
lora_attn_alpha=32
lora_dropout=0.1

# for fold in f1 f2 f3 f4 f5
for fold in full
do
    if [ ! -z "$1" ]; then
        fold=$1
    fi

    if [ ! -z "$2" ]; then
        random_seed=$2
    fi
    pre_w=$data"_"$lora_model"_"$fold"Ldi"$lora_attn_dim"La"$lora_attn_alpha"Ldr"$lora_dropout"init""$random_seed""lr""$lr"
    exp="Ps"$p_start"Hs"$hsize"Hl"$hlen"init"$random_seed"lr"$p_lr

    # # 1. make ./models/$model/weights.p (weights file) in ./models.
    echo "training"
    outdir="PRE_"$pre_w"_POST"_"$model"_"$fold""$exp""_FRZ"$freeze_lora
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
        --prefix_lr $p_lr  \
        --lora_attn_dim $lora_attn_dim \
        --lora_attn_alpha $lora_attn_alpha \
        --lora_dropout $lora_dropout \
        --ptune_start $p_start \
        --hsize $hsize \
        --hlen $hlen \
        --initial_bert_weights models/$pre_w/weights.p \
        --freeze_lora $freeze_lora \
        --msmarco True \
        --batches_per_epoch 1024 \
        # --scheduler $scheduler \
        # --warmup_epoch $warmup_epoch \
        # --weight_decay $weight_decay \
        

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
        --ptune_start $p_start \
        --hsize $hsize \
        --hlen $hlen \
        

    #3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
    echo "evaluating"
    ../bin/trec_eval -m all_trec ../data/$data/qrels models/$outdir/test.run > models/$outdir/eval.result
done
