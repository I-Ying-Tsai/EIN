
import os
import random
import torch
from transformers import BertTokenizer as bt
from transformers import BertModel as bm
from transformers import BertConfig as bc


def load_bert():
    bert_tokenizer = bt.from_pretrained('bert-base-multilingual-cased')
    bert_model = bm.from_pretrained('bert-base-multilingual-cased')

    return bert_tokenizer, bert_model


def get_sent_embedding(sent, join_sent, bert_tokenizer, bert_model, join_sents=False, mask_join_sent=False,
                       drop_mask_rate=0.0):
    sent_input_id = bert_tokenizer.encode(sent, add_special_tokens=False)
    if drop_mask_rate > 0:
        sent_input_id = drop_mask(sent_input_id, drop_mask_rate)
    input_id = [101] + sent_input_id + [102]
    
    segment_id = [0] * len(input_id)
    if join_sents:
        join_sent_input_id = bert_tokenizer.encode(join_sent, add_special_tokens=False)
        if mask_join_sent and drop_mask_rate > 0:
            join_sent_input_id = drop_mask(join_sent_input_id, drop_mask_rate)
        segment_id += [1] * (len(join_sent_input_id) + 1)
        input_id = input_id + join_sent_input_id + [102]

    # input_id = input_id[0:512]
    # segment_id = segment_id[0:512]
    outputs = bert_model(input_ids=torch.tensor([input_id]).to(bert_model.device),
                         token_type_ids=torch.tensor([segment_id]).to(bert_model.device))
    return torch.FloatTensor(outputs[0][0][0].tolist())


def drop_mask(input_id, drop_mask_rate):
    length = len(input_id)
    mask_pos = random.sample(range(length), int(length * drop_mask_rate))
    mask_pos = torch.tensor(sorted(mask_pos))
    if len(mask_pos.tolist()) != 0:
        input_id = torch.tensor(input_id)
        input_id[mask_pos] = 103
        return input_id.tolist()
    else:
        return input_id
