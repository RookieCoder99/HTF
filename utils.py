#! -*- coding:utf-8 -*-
import keras.backend as K
from keras_bert import Tokenizer
import numpy as np
import codecs
from tqdm import tqdm
import json
import unicodedata
import os
BERT_MAX_LEN = 512

class ChTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R


class HBTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
            tokens.append('[unused1]')
        return tokens

def get_tokenizer_ch(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return ChTokenizer(token_dict, cased=True)






def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)

def seq_gather(x):
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)

def extract_items(subject_model, object_model, tokenizer, text_in, id2rel, h_bar=0.5, t_bar=0.5):
    tokens = tokenizer.tokenize(text_in)
    token_ids, segment_ids = tokenizer.encode(first=text_in) 
    token_ids, segment_ids = np.array([token_ids]), np.array([segment_ids])
    if len(token_ids[0]) > BERT_MAX_LEN:
        token_ids = token_ids[:,:BERT_MAX_LEN]    
        segment_ids = segment_ids[:,:BERT_MAX_LEN] 
    sub_heads_logits, sub_tails_logits = subject_model.predict([token_ids, segment_ids])
    sub_heads, sub_tails = np.where(sub_heads_logits[0] > h_bar)[0], np.where(sub_tails_logits[0] > t_bar)[0]
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]
            # subject = tokens[sub_head: sub_tail]
            subject = tokens[sub_head: sub_tail + 1]
            subjects.append((subject, sub_head, sub_tail)) 
    if subjects:
        triple_list = []
        token_ids = np.repeat(token_ids, len(subjects), 0) 
        segment_ids = np.repeat(segment_ids, len(subjects), 0)  
        sub_heads, sub_tails = np.array([sub[1:] for sub in subjects]).T.reshape((2, -1, 1))
        obj_heads_logits, obj_tails_logits = object_model.predict([token_ids, segment_ids, sub_heads, sub_tails])
        for i, subject in enumerate(subjects):
            sub = subject[0]
            sub = ''.join([i.lstrip("##") for i in sub])
            sub = ' '.join(sub.split('[unused1]'))
            obj_heads, obj_tails = np.where(obj_heads_logits[i] > h_bar), np.where(obj_tails_logits[i] > t_bar)
            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2rel[rel_head]
                        # obj = tokens[obj_head: obj_tail]
                        obj = tokens[obj_head: obj_tail+1]
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]'))
                        triple_list.append((sub, rel, obj))
                        break
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []

def partial_match(pred_set, gold_set):
    pred = {
        (
            i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0],
            i[1],
            i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]
        )
        for i in pred_set
    }

    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold

def get_txt_list(path):
    sentences=[]
    for txt in os.listdir(path):
        with open(path+txt,"r",encoding="utf-8") as f:
            for line in f:
                sentences.append(line[:-1])
    return sentences
def extract(path,subject_model, object_model, tokenizer, id2rel, h_bar=0.5, t_bar=0.5):
    triples=[]
    sentences=get_txt_list(path)
    for sentence in sentences:
        triple=extract_items(subject_model, object_model, tokenizer, sentence, id2rel, h_bar=0.5, t_bar=0.5)
        if len(triple)!=0:
            triples.append(triple)

        # else:
            # triples.append(sentence)
    return triples


def metric(subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False, output_path='data/output/output.json'):
    if output_path:
        F = open(output_path, 'w')
    orders = ['subject', 'relation', 'object'] 
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    for line in tqdm(iter(eval_data)):
        # Pred_triples = set(extract_items(subject_model, object_model, tokenizer, line['text'], id2rel))

        Pred_triples = set(extract_items(subject_model, object_model, tokenizer, line['text'], id2rel))
        Gold_triples = set(line['triple_list'])
        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (Pred_triples, Gold_triples)

        for pred in Pred_triples_eval:
            for gold in Gold_triples_eval:
                if pred[0]==gold[0] and pred[1]==gold[1]:
                    correct_num+=1
                    break
        # correct_num += len(Pred_triples_eval & Gold_triples_eval)

        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    print('correct_num:' + str(correct_num) + '\n' + 'predict_num:' + str(predict_num) + '\ngold_num:' + str(gold_num))
    return precision, recall, f1_score
