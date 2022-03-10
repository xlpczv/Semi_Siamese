import os
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
    print(f'learning_rate nonbert={args.lr} bert={args.bert_lr}')

    ## Tensorboard
    writer = SummaryWriter(args.model_out_dir)

    ## freeze_bert
    model_name = type(model).__name__
    if(args.freeze_bert == 2):
        model.freeze_bert()

    ## parameter update setting
    nonbert_params, bert_params = model.get_params()
    optim_nonbert_params = {'params': nonbert_params, 'lr': args.lr}
    optim_bert_params = {'params': bert_params, 'lr':args.bert_lr}
    #print(nonbert_params)
    if(args.freeze_bert >= 1):
        optimizer = torch.optim.Adam([optim_nonbert_params], weight_decay=args.weight_decay)
    else: 
        optimizer = torch.optim.Adam([optim_nonbert_params, optim_bert_params], weight_decay=args.weight_decay)
    
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
        print(f'train epoch={epoch} loss={loss}')
        print(f'train epoch={epoch} loss={loss}', file=logf)
        writer.add_scalar('train_loss', loss, epoch)

        valid_score = validate(args, model, dataset, valid_run, qrelf, epoch)
        print(f'validation epoch={epoch} score={valid_score}')
        print(f'validation epoch={epoch} score={valid_score}', file=logf)
        writer.add_scalar('val_score', valid_score, epoch)

        if (top_valid_score is None) or (valid_score > top_valid_score):
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            print(f'newtopsaving epoch={epoch} score={top_valid_score}', file=logf)
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
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--grad_acc_size', type=int, default=2, help='gradient accumulation size')
    parser.add_argument('--batches_per_epoch', type=int, default=64, help='# batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate(non-bert)')
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate(bert)')
    parser.add_argument('--scheduler', type=str, default=None, help='learning rate scheduler (None/linear are  only possible)')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay rate')

    ## prompt tuning
    parser.add_argument('--hsize', type=int, default=256, help='hidden size of intermediate layer of prompt tuning')
    parser.add_argument('--hlen', type=int, default=10, help='# tokens used as prompt') 

    args = parser.parse_args()

    setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())
    
    model = MODEL_MAP[args.model](bert_model=args.model_card, hsize=args.hsize, hlen=args.hlen).cuda()

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
