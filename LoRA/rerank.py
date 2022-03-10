import argparse
import train
import data
import os
import torch


def main_cli():
    MODEL_MAP = train.modeling.MODEL_MAP
    parser = argparse.ArgumentParser('TRMD model re-ranking')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--model_card', default='bert-base-uncased', help='pretrained model card')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    parser.add_argument('--gpunum', type=str, default="0", help='gup number')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
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

    #setRandomSeed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpunum

    print("GPU count=", torch.cuda.device_count())

    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)

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
                                    
    if(args.model_weights is not None):
        wts = args.model_weights.split(',')
        if(len(wts) == 1):
            model.load(wts[0])
        elif(len(wts) == 2):
            model.load(wts[0], wts[1])

    train.run_model(args, model, dataset, run, args.out_path.name, desc='rerank')

if __name__ == '__main__':
    main_cli()
