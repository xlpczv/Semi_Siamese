#! /bin/bash
data=msmarco
model_card=bert-base-uncased
model=twinbert

random_seed=41
gpunum=0
MAX_EPOCH=10
freeze_bert=0
nblr=0.0001
bert_lr=0.00001
# scheduler=linear
# warmup_epoch=1
# weight_decay=0

# for fold in f1 f2 f3 f4 f5 ## when data is robust (Robust04) / wt (ClueWeb09b)
# for fold in f1
for fold in full ## when data is msmarco (MS-MARCO)
do
    if [ ! -z "$1" ]; then
        fold=$1
    fi

    if [ ! -z "$2" ]; then
        random_seed=$2
    fi
    exp="init""$random_seed""bert_lr"$bert_lr"nblr"$nblr

    # # 1. make ./models/$model/weights.p (weights file) in ./models.
    echo "training"
    outdir="$data"_"$model"_"$fold""$exp"
    echo $outdir

    echo $model
    python train.py \
        --model $model \
        --model_card $model_card \
        --datafiles ../data/$data/queries.tsv ../data/$data/documents.tsv \
        --qrels ../data/$data/qrels \
        --train_pairs ../data/$data/$fold.train.pairs \
        --valid_run ../data/$data/$fold.valid.run \
        --model_out_dir models/$outdir \
        --max_epoch $MAX_EPOCH \
        --gpunum $gpunum \
        --random_seed $random_seed  \
        --freeze_bert $freeze_bert \
        --bert_lr $bert_lr  \
        --non_bert_lr $nblr \
        --msmarco True \
        --batches_per_epoch 1024
        # --scheduler $scheduler \
        # --warmup_epoch $warmup_epoch \
        # --weight_decay $weight_decay
        


    # 2. load model weights from ./models/$model/weights.p, run tests, and ./models/$model/test.run
    echo "testing"
        python rerank.py \
        --model $model \
        --model_card $model_card \
        --datafiles ../data/$data/queries.tsv ../data/$data/documents.tsv \
        --run ../data/$data/$fold.test.run \
        --model_weights models/$outdir/weights.p \
        --out_path models/$outdir/test.run \
        --gpunum $gpunum \

    #3. read ./models/$model/test.run, calculate scores using various metrics and save the result to ./models/$model/eval.result
    echo "evaluating"
    ../bin/trec_eval -m all_trec ../data/$data/qrels models/$outdir/test.run > models/$outdir/eval.result
done
