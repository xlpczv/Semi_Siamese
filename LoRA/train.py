import os
import sys
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

trec_eval_f = '../bin/trec_eval'

def setRandomSeed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

def main(args, model, dataset, train_pairs, qrels, valid_run, qrelf):
    _verbose = False
    _logf = os.path.join(args.model_out_dir, 'train.log')
    print(f'learning_rate nonbert={args.non_bert_lr} bert={args.bert_lr} lora={args.lr} prefix={args.prefix_lr}')

    ## Tensorboard
    writer = SummaryWriter(args.model_out_dir)

    ## freeze_bert
    model_name = type(model).__name__
    if(args.freeze_bert == 2):
        model.freeze_bert()

    # print(model)
    ## parameter update setting
    lora_params, bert_params, non_bert_params, prefix_params = model.get_params()
    optim_lora_params = {'params': lora_params, 'lr': args.lr}
    optim_bert_params = {'params': bert_params, 'lr':args.bert_lr}
    optim_non_bert_params = {'params': non_bert_params, 'lr':args.non_bert_lr}
    optim_prefix_params = {'params': prefix_params, 'lr': args.prefix_lr}

    optim_params=[optim_non_bert_params]
    if args.freeze_bert == 0:
        optim_params.append(optim_bert_params)
        print("adding bert params to optim_params")

    if not args.freeze_lora:
        optim_params.append(optim_lora_params)
        print("adding lora params to optim_params")
    
    if not args.freeze_prefix:
        optim_params.append(optim_prefix_params)
        print("adding prefix params to optim_params")

    optimizer = torch.optim.Adam(optim_params, weight_decay=args.weight_decay)

    ## scheduler
    warmup_step = args.warmup_epoch * args.batches_per_epoch
    max_step = args.max_epoch * args.batches_per_epoch
    if args.scheduler is not None:
        print("Using linear scheduler...")
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_step, max_step, last_epoch=-1)
    else:
        print("Scheduler is None")
        scheduler = None

    ## training & validation
    logf = open(_logf, "w")
    print(f'max_epoch={args.max_epoch}', file=logf)
    epoch = 0
    top_valid_score = None
    for epoch in range(args.max_epoch):
        if args.msmarco:
            loss = train_marco_iteration(args, model, optimizer, scheduler, train_pairs)
        else:
            loss = train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels)
        print(f'train epoch={epoch} loss={loss}', file=logf)
        print(f'train epoch={epoch} loss={loss}')
        writer.add_scalar('train_loss', loss, epoch)

        valid_score = validate(args, model, dataset, valid_run, qrelf, epoch)
        print(f'validation epoch={epoch} score={valid_score}')
        print(f'validation epoch={epoch} score={valid_score}', file=logf)
        writer.add_scalar('val_score', valid_score, epoch)
        
        if (top_valid_score is None) or (valid_score > top_valid_score):
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            print(f'newtopsaving epoch={epoch} score={top_valid_score}', file=logf)
            if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
                if(args.freeze_bert >= 1):
                    model.save(os.path.join(args.model_out_dir, 'weights1.p'), os.path.join(args.model_out_dir, 'weights2.p'), without_bert=True)
                else:
                    model.save(os.path.join(args.model_out_dir, 'weights1.p'), os.path.join(args.model_out_dir, 'weights2.p'))
            else:
                if(args.freeze_bert >= 1):
                    model.save(os.path.join(args.model_out_dir, 'weights.p'), without_bert=True)
                else:
                    model.save(os.path.join(args.model_out_dir, 'weights.p'))

        logf.flush()

    print(f'topsaving score={top_valid_score}', file=logf)

def train_iteration(args, model, optimizer, scheduler, dataset, train_pairs, qrels):
    total = 0
    model.train()
    total_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=args.batch_size * args.batches_per_epoch, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, args.grad_acc_size): 
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax

            loss.backward()
            total_loss += loss.item()
            total += count
            if total % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
                    model.bert1.zero_grad()
                    model.bert2.zero_grad()
                else:
                    model.bert.zero_grad()
            pbar.update(count)
            if total >= args.batch_size * args.batches_per_epoch:
                return total_loss

def train_marco_iteration(args, model, optimizer, scheduler, train_pairs):
    total = 0
    model.train()
    total_loss = 0.
    cq_sum = 0.
    cd_sum = 0.
    with tqdm('training', total=args.batch_size * args.batches_per_epoch, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_marco_train_pairs(model, train_pairs, args.grad_acc_size): 
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax

            loss.backward()
            total_loss += loss.item()
            total += count
            if total % args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                model.zero_grad()
                if (args.model.startswith('h')) and not (args.model.startswith('hq') or args.model.startswith('hv')):
                    model.bert1.zero_grad()
                    model.bert2.zero_grad()
                else:
                    model.bert.zero_grad()
            pbar.update(count)
            if total >= args.batch_size * args.batches_per_epoch:
                return total_loss


def validate(args, model, dataset, run, qrelf, epoch):
    if args.msmarco:
        VALIDATION_METRIC = 'recip_rank'
    else:
        VALIDATION_METRIC = 'P.20'
    runf = os.path.join(args.model_out_dir, f'{epoch}.run')
    run_model(args, model, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def run_model(args, model, dataset, run, runf, desc='valid'):
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, args.batch_size):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')

def trec_eval(qrelf, runf, metric):
    print("qrelf",qrelf)
    print("runf", runf)
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def main_cli():
    MODEL_MAP = modeling.MODEL_MAP
    parser = argparse.ArgumentParser('TRMD model training and validation')
    ## model
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vbert')
    parser.add_argument('--model_card', default='bert-base-uncased', help='pretrained model card')
    parser.add_argument('--initial_bert_weights', type=str, default=None)
    parser.add_argument('--model_out_dir')

    ## data
    parser.add_argument('--msmarco', default=False, type=bool, help='whether to use ms marco or not')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))

    ## training
    parser.add_argument('--gpunum', type=str, default="0", help='gpu number')
    parser.add_argument('--random_seed', type=int, default=42, help='ranodm seed number')    
    parser.add_argument('--freeze_bert', type=int, default=0, help='freezing bert')
    parser.add_argument('--freeze_lora', type=bool, default=False, help='freezing lora')
    parser.add_argument('--freeze_prefix', type=bool, default=False, help='freezing prefix')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--grad_acc_size', type=int, default=2, help='gradient accumulation size')
    parser.add_argument('--batches_per_epoch', type=int, default=64, help='# batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate(LoRA)')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate(bert)')
    parser.add_argument('--non_bert_lr', type=float, default=1e-4, help='learning rate(non-bert)')
    parser.add_argument('--prefix_lr', type=float, default=1e-4, help='learning rate(prefix)')
    parser.add_argument('--scheduler', type=str, default=None, help='learning rate scheduler (None/linear are  only possible)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay rate')

    ## LoRA
    parser.add_argument('--lora_attn_dim', type=int, default=0, help='lora attention dimension, if 0 -> then no LoRA')
    parser.add_argument('--lora_attn_alpha', type=int, default=32, help='lora attention alpha') 
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='lora dropout rate')

    parser.add_argument('--lora_d_attn_dim', type=int, default=0, help='lora_d attention dimension, if 0 -> then no LoRA')
    parser.add_argument('--lora_d_attn_alpha', type=int, default=32, help='lora_d attention alpha') 
    parser.add_argument('--lora_d_dropout', type=float, default=0.0, help='lora_d dropout rate')

    ## prefix tuning
    parser.add_argument('--ptune_start', type=int, default=0, help='prefix tunining starting layer')
    parser.add_argument('--hsize', type=int, default=0, help='hidden size of intermediate layer of prefix tuning')
    parser.add_argument('--hlen', type=int, default=0, help='# tokens used as prefix') 

    args = parser.parse_args()

    setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())
    
    print("Load Model start")
    if args.model.startswith("hq"):
        model = MODEL_MAP[args.model.replace("hq","")](bert_model=args.model_card,
                                                    lora_attn_dim=args.lora_attn_dim, lora_attn_alpha=args.lora_attn_alpha, lora_dropout=args.lora_dropout,
                                                    lora_d_attn_dim=args.lora_d_attn_dim, lora_d_attn_alpha=args.lora_d_attn_alpha, lora_d_dropout=args.lora_d_dropout,
                                                    hetero_query=True,
                                                    p_start=args.ptune_start, hsize=args.hsize, hlen=args.hlen).cuda()
    elif args.model.startswith("hv"):
        model = MODEL_MAP[args.model.replace("hv","")](bert_model=args.model_card,
                                                    lora_attn_dim=args.lora_attn_dim, lora_attn_alpha=args.lora_attn_alpha, lora_dropout=args.lora_dropout,
                                                    lora_d_attn_dim=args.lora_d_attn_dim, lora_d_attn_alpha=args.lora_d_attn_alpha, lora_d_dropout=args.lora_d_dropout,
                                                    hetero_value=True,
                                                    p_start=args.ptune_start, hsize=args.hsize, hlen=args.hlen).cuda()
    else:
        model = MODEL_MAP[args.model](bert_model=args.model_card,
                                    lora_attn_dim=args.lora_attn_dim, lora_attn_alpha=args.lora_attn_alpha, lora_dropout=args.lora_dropout,
                                    lora_d_attn_dim=args.lora_d_attn_dim, lora_d_attn_alpha=args.lora_d_attn_alpha, lora_d_dropout=args.lora_d_dropout,
                                    p_start=args.ptune_start, hsize=args.hsize, hlen=args.hlen).cuda()
    print("Load Model end")

    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    if args.msmarco:
        train_pairs = args.train_pairs
    else:
        train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    ## initial
    if(args.initial_bert_weights is not None):
        wts = args.initial_bert_weights.split(',')
        if(len(wts) == 1):
            model.load(wts[0])
        elif(len(wts) == 2):
            model.load_duet(wts[0], wts[1])

    os.makedirs(args.model_out_dir, exist_ok=True)
    main(args, model, dataset, train_pairs, qrels, valid_run, args.qrels.name)


if __name__ == '__main__':
    main_cli()
