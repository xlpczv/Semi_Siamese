from pytools import memoize_method
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import modeling_util
import string

class BertRanker(torch.nn.Module):
    def __init__(self, without_bert=False, bert_model=None):
        super().__init__()
        self.BERT_MODEL = bert_model
        if "bert-base" in self.BERT_MODEL:
            self.CHANNELS = 12 + 1 # from bert-base-uncased
            self.BERT_SIZE = 768 # from bert-base-uncased
        elif "bert-large" in self.BERT_MODEL:
            self.CHANNELS = 24 + 1 # from bert-base-uncased
            self.BERT_SIZE = 1024 # from bert-base-uncased

        if(without_bert):
            self.bert = bert_model
        else:
            self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path, without_bert=False):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            state[key] = state[key].data
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def load_cuda(self, path, device):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
        print("load model set device : ", path, device)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        print("non_bert_params", [k for k, v in params if not k.startswith('bert')])
        print("bert_params", [k for k, v in params if k.startswith('bert')])
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask):
        BATCH, QLEN = query_tok.shape
        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        #result = self.bert(toks, segment_ids.long(), mask)
        result_tuple = self.bert(toks, mask, segment_ids.long())
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1:QLEN+1] for r in result]
        doc_results = [r[:, QLEN+2:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask, device='cuda:0'):
        # encode without subbatching
        BATCH, QLEN = query_tok.shape
        DIFF = 5 # = [CLS], 2x[SEP], [Q], [D]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - DIFF

        query_toks = query_tok
        # query_mask = query_mask
        doc_toks = doc_tok[:, :MAX_DOC_TOK_LEN]
        doc_mask = doc_mask[:, :MAX_DOC_TOK_LEN]
        
        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        Q_tok = torch.full(
            size=(BATCH, 1), fill_value=1, dtype=torch.long
        ).cuda(device)  # [unused0] = 1
        D_tok = torch.full(
            size=(BATCH, 1), fill_value=2, dtype=torch.long
        ).cuda(device)  # [unused1] = 2

        # Query augmentation with [MASK] tokens ([MASK] = 103)
        query_toks[query_toks == -1] = torch.tensor(103).cuda(device)
        query_mask = torch.ones_like(query_mask)

        # build BERT input sequences
        toks = torch.cat([CLSS, Q_tok, query_toks, SEPS, D_tok, doc_toks, SEPS], dim=1)
        mask = torch.cat([ONES, ONES, query_mask, ONES, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (3+QLEN) + [ONES] * (2+doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)
        
        # modifiy doc_mask
        doc_mask = torch.cat([ONES, doc_mask, ONES], dim=1)

        # execute BERT model
        result_tuple = self.bert(toks, mask, segment_ids.long())
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:, :QLEN+3] for r in result]
        doc_results = [r[:, QLEN+3:] for r in result]

        cls_results = [r[:, 0] for r in result]

        return cls_results, query_results, query_mask, doc_results, doc_mask

class PBertRanker(torch.nn.Module):
    def __init__(self, without_bert=False, bert_model=None):
        super().__init__()
        self.BERT_MODEL = bert_model
        if "bert-base" in self.BERT_MODEL:
            self.CHANNELS = 12 + 1 # from bert-base-uncased
            self.BERT_SIZE = 768 # from bert-base-uncased
        elif "bert-large" in self.BERT_MODEL:
            self.CHANNELS = 24 + 1 # from bert-base-uncased
            self.BERT_SIZE = 1024 # from bert-base-uncased

        if(without_bert):
            self.bert = bert_model
        else:
            self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path, without_bert=False):
        if(without_bert):
            state_org = self.state_dict(keep_vars=True)
            state = {}
            for key in list(state_org):
                if('bert.' in key): continue
                state[key] = state_org[key].data
        else:
            state = self.state_dict(keep_vars=True)
            for key in list(state):
                state[key] = state[key].data

        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def load_cuda(self, path, device):
        self.load_state_dict(torch.load(path, map_location=torch.device(device)), strict=False)
        print("load model set device : ", path, device)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        print("non_bert_params", [k for k, v in params if not k.startswith('bert')])
        print("bert_params", [k for k, v in params if k.startswith('bert')])
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask, p_embeds=None):
        BATCH, QLEN = query_tok.shape
        PROMPT = p_embeds.shape[0]
        #QLEN = QLEN + PROMPT

        prompt_tok = torch.full(
            size=(BATCH, PROMPT), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        prompt_mask = torch.full(
            size=(BATCH, PROMPT), fill_value=1., dtype=torch.float
        ).cuda()  

        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings

        MAX_DOC_TOK_LEN = maxlen - QLEN - PROMPT - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        prompt_toks = torch.cat([prompt_tok] * sbcount, dim=0)
        prompt_mask = torch.cat([prompt_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences
        toks = torch.cat([prompt_toks, CLSS, query_toks, SEPS, doc_toks, SEPS], dim=1)
        mask = torch.cat([prompt_mask, ONES, query_mask, ONES, doc_mask, ONES], dim=1)
        segment_ids = torch.cat([NILS] * (2 + QLEN + PROMPT) + [ONES] * (1 + doc_toks.shape[1]), dim=1)
        toks[toks == -1] = 0 # remove padding (will be masked anyway)

        # execute BERT model
        #result_tuple = self.bert(toks, mask, segment_ids.long())
        result_tuple = self.bert(toks, mask, segment_ids.long(), prompt_inputs_embeds=p_embeds)
        result = result_tuple[2] ## all hidden_states

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1+PROMPT:QLEN+1] for r in result]
        doc_results = [r[:, QLEN+2+PROMPT:-1] for r in result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        cls_results = []
        for layer in result:
            cls_output = layer[:, 0+PROMPT]  ## important
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            cls_results.append(cls_result)

        return cls_results, query_results, doc_results


class TwoBertRanker(torch.nn.Module):
    def __init__(self, without_bert=False, asym=False, bert_model=None):
        super().__init__()
        self.BERT_MODEL = bert_model
        if "bert-base" in self.BERT_MODEL:
            self.CHANNELS = 12 + 1 # from bert-base-uncased
            self.BERT_SIZE = 768 # from bert-base-uncased
        elif "bert-large" in self.BERT_MODEL:
            self.CHANNELS = 24 + 1 # from bert-base-uncased
            self.BERT_SIZE = 1024 # from bert-base-uncased

        if(without_bert): 
            self.bert = None
        else:
            self.bert = BertModel.from_pretrained(self.BERT_MODEL, output_hidden_states=True)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)

    def forward(self, **inputs):
        raise NotImplementedError

    def save(self, path, without_bert=False):
        if(without_bert):
            state_org = self.state_dict(keep_vars=True)
            state = {}
            for key in list(state_org):
                if('bert.' in key): continue
                state[key] = state_org[key].data
        else:
            state = self.state_dict(keep_vars=True)
            for key in list(state):
                state[key] = state[key].data

        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
        print("load model : ", path)

    def freeze_bert(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def get_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        non_bert_params = [v for k, v in params if not k.startswith('bert')]
        bert_params = [v for k, v in params if k.startswith('bert')]
        print("non_bert_params", [k for k, v in params if not k.startswith('bert')])
        print("bert_params", [k for k, v in params if k.startswith('bert')])
        return non_bert_params, bert_params 

    @memoize_method
    def tokenize(self, text):
        toks = self.tokenizer.tokenize(text)
        toks = [self.tokenizer.vocab[t] for t in toks]
        return toks

    def encode_bert(self, query_tok, query_mask, doc_tok, doc_mask, p_qrepr=None, p_drepr=None):
        BATCH, QLEN = query_tok.shape
        assert(p_qrepr.shape == p_drepr.shape)
        PROMPT = p_qrepr.shape[0]

        prompt_tok = torch.full(
            size=(BATCH, PROMPT), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        prompt_mask = torch.full(
            size=(BATCH, PROMPT), fill_value=1., dtype=torch.float
        ).cuda()  

        DIFF = 3 # = [CLS] and 2x[SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - QLEN - PROMPT - DIFF

        doc_toks, sbcount = modeling_util.subbatch(doc_tok, MAX_DOC_TOK_LEN)
        doc_mask, _ = modeling_util.subbatch(doc_mask, MAX_DOC_TOK_LEN)

        query_toks = torch.cat([query_tok] * sbcount, dim=0)
        query_mask = torch.cat([query_mask] * sbcount, dim=0)

        prompt_toks = torch.cat([prompt_tok] * sbcount, dim=0)
        prompt_mask = torch.cat([prompt_mask] * sbcount, dim=0)

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[CLS]'])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab['[SEP]'])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # build BERT input sequences query & doc
        q_toks = torch.cat([prompt_toks, CLSS, query_toks, SEPS], dim=1)
        q_mask = torch.cat([prompt_mask, ONES, query_mask, ONES], dim=1)
        q_segid = torch.cat([NILS] * (2+QLEN+PROMPT), dim=1)
        q_toks[q_toks == -1] = 0

        d_toks = torch.cat([prompt_toks, CLSS, doc_toks, SEPS], dim=1)
        d_mask = torch.cat([prompt_mask, ONES, doc_mask, ONES], dim=1)
        d_segid = torch.cat([NILS] * (2+doc_toks.shape[1]+PROMPT), dim=1)
        d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long(), prompt_inputs_embeds=p_qrepr)
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long(), prompt_inputs_embeds=p_drepr)
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        query_results = [r[:BATCH, 1+PROMPT:-1] for r in q_result]
        doc_results = [r[:, 1+PROMPT:-1] for r in d_result]
        doc_results = [modeling_util.un_subbatch(r, doc_tok, MAX_DOC_TOK_LEN) for r in doc_results]

        # build CLS representation
        q_cls_results = []
        for layer in q_result:
            cls_output = layer[:, 0+PROMPT]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            q_cls_results.append(cls_result)

        d_cls_results = []
        for layer in d_result:
            cls_output = layer[:, 0+PROMPT]
            cls_result = []
            for i in range(cls_output.shape[0] // BATCH):
                cls_result.append(cls_output[i*BATCH:(i+1)*BATCH])
            cls_result = torch.stack(cls_result, dim=2).mean(dim=2)
            d_cls_results.append(cls_result)

        return q_cls_results, d_cls_results, query_results, doc_results

    def encode_colbert(self, query_tok, query_mask, doc_tok, doc_mask, device='cuda:0', p_qrepr=None, p_drepr=None):
        # encode without subbatching
        query_lengths = (query_mask > 0).sum(1)
        doc_lengths = (doc_mask > 0).sum(1)
        BATCH, QLEN = query_tok.shape
        PROMPT = p_qrepr.shape[0]
        # QLEN : 20
        DIFF = 2  # = [CLS] and [SEP]
        maxlen = self.bert.config.max_position_embeddings
        MAX_DOC_TOK_LEN = maxlen - PROMPT - DIFF  # doc maxlen: 510

        prompt_tok = torch.full(
            size=(BATCH, PROMPT), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        prompt_mask = torch.full(
            size=(BATCH, PROMPT), fill_value=1., dtype=torch.float
        ).cuda()  

        doc_toks = F.pad(doc_tok[:, : MAX_DOC_TOK_LEN], pad=(0, 1, 0, 0), value=-1)
        doc_mask = F.pad(doc_mask[:, : MAX_DOC_TOK_LEN], pad=(0, 1, 0, 0), value=0)
        query_toks = query_tok

        query_lengths = torch.where(query_lengths > 19, torch.tensor(19).cuda(device), query_lengths)
        query_toks[torch.arange(BATCH), query_lengths] = self.tokenizer.vocab["[SEP]"]
        query_mask[torch.arange(BATCH), query_lengths] = 1
        #doc_lengths = torch.where(doc_lengths > 510, torch.tensor(510).cuda(device), doc_lengths)
        doc_lengths = torch.where(doc_lengths > MAX_DOC_TOK_LEN, torch.tensor(MAX_DOC_TOK_LEN).cuda(device), doc_lengths)
        doc_toks[torch.arange(BATCH), doc_lengths] = self.tokenizer.vocab["[SEP]"]
        doc_mask[torch.arange(BATCH), doc_lengths] = 1

        CLSS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[CLS]"])
        SEPS = torch.full_like(query_toks[:, :1], self.tokenizer.vocab["[SEP]"])
        ONES = torch.ones_like(query_mask[:, :1])
        NILS = torch.zeros_like(query_mask[:, :1])

        # print("prompt_tok.shape", prompt_tok.shape, "query_toks.shape", query_toks.shape)
        # print("prompt_masks.ahep", prompt_mask.shape, "query_mask.shape", query_mask.shape)
        # build BERT input sequences query & doc
        q_toks = torch.cat([prompt_tok, CLSS, query_toks], dim=1)
        q_mask = torch.cat([prompt_mask, ONES, query_mask], dim=1)
        q_segid = torch.cat([NILS] * (1 + QLEN + PROMPT), dim=1)
        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_toks[q_toks == -1] = torch.tensor(103).cuda(device)

        d_toks = torch.cat([prompt_tok, CLSS, doc_toks], dim=1)
        d_mask = torch.cat([prompt_mask, ONES, doc_mask], dim=1)
        d_segid = torch.cat([NILS] * (1 + doc_toks.shape[1] + PROMPT), dim=1)
        d_toks[d_toks == -1] = 0

        # execute BERT model
        q_result_tuple = self.bert(q_toks, q_mask, q_segid.long(), prompt_inputs_embeds=p_qrepr)
        d_result_tuple = self.bert(d_toks, d_mask, d_segid.long(), prompt_inputs_embeds=p_drepr)
        q_result = q_result_tuple[2]
        d_result = d_result_tuple[2]

        # extract relevant subsequences for query and doc
        # query_results = [r[:, PROMPT:] for r in q_result]  # missing representation for cls and sep?
        # doc_results = [r[:, PROMPT:] for r in d_result]
        query_results = [r[:, :] for r in q_result]  # missing representation for cls and sep?
        doc_results = [r[:, :] for r in d_result]

        q_cls_result = [r[:, PROMPT] for r in q_result]
        d_cls_result = [r[:, PROMPT] for r in d_result]

        return q_cls_result, d_cls_result, query_results, q_mask, doc_results, d_mask, prompt_tok

class VanillaBertRanker(BertRanker):
    def __init__(self, without_bert=False, bert_model=None):
        super().__init__(without_bert=without_bert, bert_model=bert_model)
        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask)
        if(value_return):
            return self.cls(self.dropout(cls_reps[-1])), cls_reps, None
        else:
            return self.cls(self.dropout(cls_reps[-1]))
    
    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        return self.cls(self.dropout(cls_reps[-1])), None

class PMonoBertRanker(PBertRanker):
    def __init__(self, without_bert=False, bert_model=None, hsize=256, hlen=10):
        super().__init__(without_bert=without_bert, bert_model=bert_model)
        self.hsize = hsize  ## prompt hidden size
        self.hlen = hlen    ## prompt length

        self.rep_l1 = torch.nn.Linear(self.BERT_SIZE, self.hsize)
        self.rep_relu = torch.nn.ReLU()
        self.rep_l2 = torch.nn.Linear(self.hsize, self.BERT_SIZE)

        self.dropout = torch.nn.Dropout(0.1)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

        theta_val = self.initialize_theta()
        self.rep_theta = torch.nn.Parameter(theta_val, requires_grad=True) ## channel * prompt_len * BERT_SIZE

    def initialize_theta(self):  ## similia shape 
        with torch.no_grad():
            vocab_rep = self.bert.embeddings.word_embeddings.weight
            vocab_rep_mean = torch.mean(vocab_rep)
            vocab_rep_std = torch.std(vocab_rep)
            theta_reps = torch.normal(vocab_rep_mean, vocab_rep_std, size=(self.hlen, self.BERT_SIZE))
            print("initialize mean, std : ", vocab_rep_mean, vocab_rep_std)
            print("initialize shape : ", theta_reps.shape)
            #print(theta_reps)
            return theta_reps

    def get_theta(self):
        with torch.no_grad():
            theta_mid = self.rep_l2(self.rep_relu(self.rep_l1(self.rep_theta)))  
            theta = theta_mid.reshape(self.hlen, -1, self.BERT_SIZE).permute(1,0,2) 
            return theta

    def get_ranker_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        bert_params = []
        non_bert_params = [] 
        for k, v in params:
            if('bert' in k): continue
            if('rep_' in k): continue
            non_bert_params.append(v)

        return non_bert_params, bert_params 

    def get_prefix_params(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        bert_params = []
        non_bert_params = [] 
        for k, v in params:
            if('rep_' in k): 
                non_bert_params.append(v)

        return non_bert_params, bert_params 

    def freeze_prefix(self):
        params = [(k, v) for k, v in self.named_parameters() if v.requires_grad]
        for k, v in params:
            if('rep_' in k): 
                v.requires_grad=False

    def put_theta(self, theta):
        weight = self.bert.embeddings.word_embeddings.weight
        weight[1:1+self.hlen] = theta

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        theta = self.rep_l2(self.rep_relu(self.rep_l1(self.rep_theta)))  ## hlen * BERT_SIZE
        cls_reps, _, _ = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask, p_embeds=theta) 
        if(value_return):
            return self.cls(self.dropout(cls_reps[-1])), cls_reps, None
        else:
            return self.cls(self.dropout(cls_reps[-1]))

    def forward_with_cls(self, cls_reps):
        return self.cls(self.dropout(cls_reps[-1]))

class P0TwinBertRanker(TwoBertRanker):
    def __init__(self, without_bert=False, qd=True, asym=False, bert_model=None, hsize=256, hlen=10):
        super().__init__(without_bert=without_bert, asym=asym, bert_model=bert_model)
        self.qd = qd
        self.dropout = torch.nn.Dropout(0.1)
        self.wpool = torch.nn.AdaptiveAvgPool2d((1,self.BERT_SIZE))
        self.res = torch.nn.Linear(self.BERT_SIZE, self.BERT_SIZE)
        self.cls = torch.nn.Linear(self.BERT_SIZE, 1)

        self.hsize = hsize  ## prompt hidden size
        self.hlen = hlen    ## prompt length

        self.rep_l1 = torch.nn.Linear(self.BERT_SIZE, self.hsize)
        self.rep_relu = torch.nn.ReLU()
        self.rep_l2 = torch.nn.Linear(self.hsize, self.BERT_SIZE)

        theta_val = self.initialize_theta()
        self.rep_theta = torch.nn.Parameter(theta_val, requires_grad=True) ## channel * prompt_len * BERT_SIZE

    def initialize_theta(self):  ## similia shape
        with torch.no_grad():
            vocab_rep = self.bert.embeddings.word_embeddings.weight
            vocab_rep_mean = torch.mean(vocab_rep)
            vocab_rep_std = torch.std(vocab_rep)
            theta_reps = torch.normal(vocab_rep_mean, vocab_rep_std, size=(self.hlen, self.BERT_SIZE))
            print("initialize mean, std : ", vocab_rep_mean, vocab_rep_std)
            print("initialize shape : ", theta_reps.shape)
            #print(theta_reps)
            return theta_reps

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        theta = self.rep_l2(self.rep_relu(self.rep_l1(self.rep_theta)))

        q_cls_reps, d_cls_reps, q_reps, d_reps = self.encode_bert(query_tok, query_mask, doc_tok, doc_mask,
            p_qrepr=theta,
            p_drepr=theta
        )

        if(self.qd):
            x1 = self.wpool(q_reps[-1]).squeeze(dim=1)
            x2 = self.wpool(d_reps[-1]).squeeze(dim=1)
        else:
            x1 = q_cls_reps[-1]
            x2 = d_cls_reps[-1]

        x = torch.max(x1, x2)
        feature = self.res(x)+x
        score = self.cls(feature)

        simmat = feature					## for distillation ( features from represenation)
        #simmat = torch.cat([q_reps[-1], d_reps[-1]], dim=1)	## for distillation ( representation )
        if(value_return):
            if(self.qd):
                return score, None, simmat 
            else:
                return score, simmat, None
        else:
            return score

    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        x1 = self.wpool(q_reps[-1]).squeeze(dim=1)
        x2 = self.wpool(d_reps[-1]).squeeze(dim=1) 

        x = torch.max(x1, x2)
        feature = self.res(x)+x
        score = self.cls(feature)

        simmat = feature
        #simmat = torch.cat([q_reps[-1], d_reps[-1]], dim=1)	## for distilation
        return score, simmat

class P0ColBertRanker(TwoBertRanker):
    def __init__(self, without_bert=False, asym=False, bert_model=None, hsize=256, hlen=10):
        super().__init__(without_bert=without_bert, asym=asym, bert_model=bert_model)
        self.dim = 128  # default: dim=128
        self.skiplist = self.tokenize(string.punctuation)

        self.clinear = torch.nn.Linear(
            self.BERT_SIZE, self.dim, bias=False
        )  # both for queries, documents

        ## prompt
        self.hsize = hsize  ## prompt hidden size
        self.hlen = hlen    ## prompt length

        self.rep_l1 = torch.nn.Linear(self.BERT_SIZE, self.hsize)
        self.rep_relu = torch.nn.ReLU()
        self.rep_l2 = torch.nn.Linear(self.hsize, self.BERT_SIZE)

        theta_val = self.initialize_theta()
        self.rep_theta = torch.nn.Parameter(theta_val, requires_grad=True) ## channel * prompt_len * BERT_SIZE

    def initialize_theta(self):  ## similia shape 
        with torch.no_grad():
            vocab_rep = self.bert.embeddings.word_embeddings.weight
            vocab_rep_mean = torch.mean(vocab_rep)
            vocab_rep_std = torch.std(vocab_rep)
            theta_reps = torch.normal(vocab_rep_mean, vocab_rep_std, size=(self.hlen, self.BERT_SIZE))
            print("initialize mean, std : ", vocab_rep_mean, vocab_rep_std)
            print("initialize shape : ", theta_reps.shape)
            #print(theta_reps)
            return theta_reps

    def forward(self, query_tok, query_mask, doc_tok, doc_mask, value_return=False):
        ## prompt
        theta = self.rep_l2(self.rep_relu(self.rep_l1(self.rep_theta)))

        # q length default: 32  -> 20
        # d length defualt: 180 -> 510

        # 1) Prepend [Q] token to query, [D] token to document
        q_length = query_tok.shape[1]
        d_length = doc_tok.shape[1]
        num_batch_samples = doc_tok.shape[0]

        Q_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=1, dtype=torch.long
        ).cuda()  # [unused0] = 1
        D_tok = torch.full(
            size=(num_batch_samples, 1), fill_value=2, dtype=torch.long
        ).cuda()  # [unused1] = 2
        one_tok = torch.full(size=(num_batch_samples, 1), fill_value=1).cuda()

        query_tok = torch.cat([Q_tok, query_tok[:, : q_length - 1]], dim=1)
        doc_tok = torch.cat([D_tok, doc_tok[:, : d_length - 1]], dim=1)
        query_mask = torch.cat([one_tok, query_mask[:, : q_length - 1]], dim=1)
        doc_mask = torch.cat([one_tok, doc_mask[:, : d_length - 1]], dim=1)

        # 2) Query augmentation with [MASK] tokens ([MASK] = 103)
        q_cls_reps, d_cls_reps, q_reps, query_mask, d_reps, doc_mask, prompt_tok = self.encode_colbert( ## prompt_tok.shape=[bs, hlen]
            query_tok, query_mask, doc_tok, doc_mask, 
            p_qrepr=theta,
            p_drepr=theta
        )  # reps includes rep of [CLS], [SEP]
        # print("prompt_tok", prompt_tok)

        col_q_reps = self.clinear(q_reps[-1])
        col_d_reps = self.clinear(d_reps[-1])

        # 3) skip punctuations in doc tokens
        # print("one_tok.shape", one_tok.shape, "doc_tok.shape", doc_tok.shape) # [bs, 1] [bs, 800]
        # cut_doc_tok = torch.cat([one_tok.long(), doc_tok[:, :510], one_tok.long()], dim=1)
        cut_doc_tok = torch.cat([one_tok.long(), prompt_tok, doc_tok[:, :510-self.hlen], one_tok.long()], dim=1) ## Euna changed
        # print("cut_doc_tok.shape", cut_doc_tok.shape, "doc_mask.shape", doc_mask.shape) # [bs, 512] [bs, 512]
        mask = torch.ones_like(doc_mask, dtype=torch.float).cuda()
        mask = torch.where(
            ((cut_doc_tok >= 999) & (cut_doc_tok <= 1013))
            | ((cut_doc_tok >= 1024) & (cut_doc_tok <= 1036))
            | ((cut_doc_tok >= 1063) & (cut_doc_tok <= 1066))
            | (cut_doc_tok == -1),
            torch.tensor(0.0).cuda(),
            doc_mask,
        )
        # print("col_d_reps.shape", col_d_reps.shape, "doc_mask.shape", doc_mask.shape, "mask.shape", mask.shape)
        col_d_reps = col_d_reps * mask.unsqueeze(2)
        q_rep = F.normalize(col_q_reps, p=2, dim=2)
        d_rep = F.normalize(col_d_reps, p=2, dim=2)
        score = (q_rep @ d_rep.permute(0, 2, 1)).max(2).values.sum(1)
        #simmat = (q_rep @ d_rep.permute(0, 2, 1))
        #score = simmat.max(2).values.sum(1)
        simmat = torch.cat([q_rep, d_rep], dim=1)  ## for distillation
        score = score.unsqueeze(1)
        if(value_return):
            return score, None, simmat
        else:
            return score

    def forward_without_bert(self, cls_reps, q_reps, d_reps, query_tok, doc_tok):
        score = (q_reps @ d_reps.permute(0, 2, 1)).max(2).values.sum(1)
        simmat = torch.cat([q_reps, d_reps], dim=1)  ## for distillation
        score = score.unsqueeze(1)
        return score, simmat


MODEL_MAP = {
    'pmonobert' : PMonoBertRanker,
    'p0colbert' : P0ColBertRanker,
    'p0twinbert': P0TwinBertRanker,
}