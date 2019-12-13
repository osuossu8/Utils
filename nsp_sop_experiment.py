
'''
# Mecab setup in GoogleColab

!apt install curl mecab libmecab-dev mecab-ipadic-utf8 file -y
!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
%cd mecab-ipadic-neologd
!bin/install-mecab-ipadic-neologd -n -a -y --prefix /var/lib/mecab/dic/mecab-ipadic-neologd
!sed -i -e 's@^dicdir.*$@dicdir = /var/lib/mecab/dic/mecab-ipadic-neologd@' /etc/mecabrc
!pip install mecab-python3
%cd ../
!ls

# transformer
!pip install transformers > /dev/null

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from collections import OrderedDict
import collections
import datetime
import pkg_resources
import seaborn as sns
import time
import scipy.stats as stats
import gc
import random
import re
import operator 
import six
import sys
from sklearn import metrics
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
%load_ext autoreload
%autoreload 2
%matplotlib inline
from tqdm import tqdm, tqdm_notebook
import os
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
# from apex import amp
import shutil
from transformers import (WEIGHTS_NAME, BertConfig, BertModel, PreTrainedModel, BertPreTrainedModel,
                          AlbertModel, AlbertForSequenceClassification, 
                          AlbertTokenizer, AlbertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer, DistilBertTokenizer, DistilBertModel)

from transformers import AdamW # , WarmupLinearSchedule
from transformers.tokenization_bert import (BasicTokenizer,
                                                    whitespace_tokenize)

import MeCab

class MecabTokenizer:
    def __init__(self):
        self.wakati = MeCab.Tagger('-Owakati')
        self.wakati.parse('')

    def tokenize(self, line):
        txt = self.wakati.parse(line)
        txt = txt.split()
        return txt


class TokenizerForBertJP(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):

        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.tokenizer = MecabTokenizer()
        self.do_lower_case = do_lower_case
        self.max_len = max_len if max_len is not None else int(1e12)

    def get_padding_idx(self):
        return self.vocab['[PAD]']

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self.convert_to_unicode(text)
        output_tokens = self.tokenizer.tokenize(text)
        return output_tokens

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = OrderedDict()
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = self.convert_to_unicode(reader.readline())
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of [tokens|ids] using the vocab."""
        output = []
        for item in tokens:
            if item in self.vocab:
                output.append(self.vocab[item])
            else:
                output.append(self.vocab['[UNK]'])
        if len(output) > self.max_len:
            raise ValueError(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return output


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join([x for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


# ==================
# create SOP dataset
# ==================

def write_instance_to_example_files(instances, tokenizer, max_seq_length):

    return_df = pd.DataFrame(columns = ['input_ids', 'input_mask', 'segment_ids'])
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_next_sentence_labels = []
    for (inst_index, instance) in tqdm(enumerate(instances)):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        sentence_order_label = 1 if instance.is_random_next else 0

        # Note: We keep this feature name `next_sentence_labels` to be compatible
        # with the original data created by lanzhzh@. However, in the ALBERT case
        # it does contain sentence_order_label.

        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)
        all_next_sentence_labels.append(sentence_order_label)

    return_df['input_ids'] = all_input_ids
    return_df['input_mask'] = all_input_mask
    return_df['segment_ids'] = all_segment_ids
    return_df['sentence_order_label'] = all_next_sentence_labels
    return return_df


def create_training_instances(df, tokenizer, max_seq_length, dupe_factor, short_seq_prob, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  
  for label in tqdm(range(df["label"].nunique())):
      tmp_docs = []
      for line in df[df["label"]==label]["news"].values.tolist():
          # Empty lines are used as document delimiters
          #if not line:
          #    all_documents.append([])
          tokens = tokenizer.tokenize(line)
          tmp_docs.append(tokens)
      all_documents.append(tmp_docs)
      all_documents.append([])
  
  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  instances = []
  for _ in range(dupe_factor):
      for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_length, short_seq_prob, rng))

  rng.shuffle(instances)
  return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or \
            (FLAGS.random_next_sentence and rng.random() < 0.5):
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        elif not FLAGS.random_next_sentence and rng.random() < 0.5:
          is_random_next = True
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
          # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
          tokens_a, tokens_b = tokens_b, tokens_a
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances

class FLAGS:
    max_seq_length = 256 # 512
    short_seq_prob = 0.1
    random_seed = 42
    random_next_sentence = False
    dupe_factor = 40
    masked_lm_prob = 0.15
    max_predictions_per_seq = 20

rng = random.Random(FLAGS.random_seed)
BERT_MODEL_PATH = '/your/path/to/jp_bert_model/'

vocab_file_path = os.path.join(BERT_MODEL_PATH, 'vocab.txt')

tokenizer = TokenizerForBertJP(vocab_file_path)

instances = create_training_instances(df, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor, FLAGS.short_seq_prob, rng)
features_df = write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length)

class DownSampler(object):
    def __init__(self, random_states):
        self.random_states = random_states

    def transform(self, data, target):
        positive_data = data[data[target] == 0]
        positive_ratio = len(positive_data) / len(data)
        negative_data = data[data[target] == 1].sample(
            frac=positive_ratio / (1 - positive_ratio), random_state=self.random_states)
        return pd.concat([positive_data, negative_data], sort=True).reset_index()

ds = DownSampler(FLAGS.random_seed)

down_sampled_df = ds.transform(features_df, 'sentence_order_label')


# ==================
# create NSP dataset
# ==================

def create_nsp_training_instances(df, tokenizer, max_seq_length, dupe_factor, short_seq_prob, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  
  for label in tqdm(range(df["label"].nunique())):
      tmp_docs = []
      for line in df[df["label"]==label]["news"].values.tolist():
          # Empty lines are used as document delimiters
          #if not line:
          #    all_documents.append([])
          tokens = tokenizer.tokenize(line)
          tmp_docs.append(tokens)
      all_documents.append(tmp_docs)
      all_documents.append([])
  
  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  instances = []
  for _ in range(dupe_factor):
      for document_index in range(len(all_documents)):
        instances.extend(
            create_nsp_instances_from_document(
                all_documents, document_index, max_seq_length, short_seq_prob, rng))

  rng.shuffle(instances)
  return instances

def create_nsp_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances

nsp_instances = create_nsp_training_instances(df, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor, FLAGS.short_seq_prob, rng)
nsp_features_df = write_instance_to_example_files(nsp_instances, tokenizer, FLAGS.max_seq_length)

nsp_down_sampled_df = ds.transform(nsp_features_df, 'sentence_order_label')

class CustomNSPHead(nn.Module):
    def __init__(self, config):
        super(CustomNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 1)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score

class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = CustomNSPHead(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)
        return seq_relationship_score

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import (Dataset,DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

TARGET_COLUMNS = 'sentence_order_label'

y = sop_df[TARGET_COLUMNS].values 
# y = nsp_df[TARGET_COLUMNS].values 
enc = OneHotEncoder(categories="auto", sparse=False, dtype=np.float32)
onehot_y = enc.fit_transform(y.reshape(-1, 1))

SEED = 1129
fold_id = 0
num_folds = 5
epochs = 5
batch_size = 2
device = 'cuda'

# train_idx, test_idx = train_test_split(list(nsp_df.index), test_size=0.2, random_state=SEED, stratify=y)
# train_idx, val_idx = train_test_split(list(nsp_df.iloc[train_idx].index), test_size=0.2, random_state=SEED, stratify=y[train_idx])

train_idx, test_idx = train_test_split(list(sop_df.index), test_size=0.2, random_state=SEED, stratify=y)
train_idx, val_idx = train_test_split(list(sop_df.iloc[train_idx].index), test_size=0.2, random_state=SEED, stratify=y[train_idx])

y_train = y[train_idx]
y_val = y[val_idx]

features = [
    torch.tensor(np.array([i for i in nsp_df['input_ids'].values])[train_idx].astype("int32"), dtype=torch.long),
    torch.tensor(np.array([i for i in nsp_df['input_mask'].values])[train_idx].astype("int32"), dtype=torch.long)
]

val_features = [
    torch.tensor(np.array([i for i in nsp_df['input_ids'].values])[val_idx].astype("int32"), dtype=torch.long),
    torch.tensor(np.array([i for i in nsp_df['input_mask'].values])[val_idx].astype("int32"), dtype=torch.long)
]

y_train_torch = torch.tensor(y_train, dtype=torch.float32)
y_val_torch = torch.tensor(y_val, dtype=torch.float32)

train_dataset = TensorDataset(*features, y_train_torch)
val_dataset = TensorDataset(*val_features, y_val_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

BERT_MODEL_PATH = '/content/drive/My Drive/PyTorch版/'
bert_config = BertConfig(BERT_MODEL_PATH+'bert_config.json')
bert_config.layer_norm_eps=1e-12
bert_config.num_hidden_layers = 6
model = BertForNextSentencePrediction(bert_config)
model.to(device)

lr = 1e-5
criterion = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_one_epoch(model, train_loader, criterion, optimizer, device, steps_upd_logging=500, accumulation_steps=1,
                                 multi_loss=None):
    model.train()

    total_loss = 0.0
    for step, (input_ids, input_mask, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(input_ids, input_mask)
        logits = torch.squeeze(logits)

        if multi_loss is not None:
            logits, logits_l2 = logits
            l2_loss = multi_loss(logits_l2, targets)
        loss = criterion(logits, targets)
        if multi_loss is not None:
            loss = l2_loss*0.5 + l2_loss*0.5
        loss.backward()

        if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
            optimizer.step()  # Now we can do an optimizer step

        total_loss += loss.item()

        if (step + 1) % steps_upd_logging == 0:
            print('Train loss on step {} was {}'.format(step + 1, round(total_loss / (step + 1), 5)))


    return total_loss / (step + 1)


def validate(model, val_loader, criterion, device, multi_loss=None):
    model.eval()

    val_loss = 0.0
    true_ans_list = []
    preds_cat = []
    for step, (input_ids, input_mask, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        targets = targets.to(device)

        logits = model(input_ids, input_mask)
        logits = torch.squeeze(logits)
        if multi_loss is not None:
            logits, logits_l2 = logits
            l2_loss = multi_loss(logits_l2, targets)
        loss = criterion(logits, targets)
        if multi_loss is not None:
            loss = l2_loss * 0.5 + l2_loss * 0.5
        val_loss += loss.item()

        targets = targets.float().cpu().detach().numpy()
        logits = torch.sigmoid(logits).float().cpu().detach().numpy().astype("float32")
        if multi_loss:
            logits_l2 = logits_l2.float().cpu().detach().numpy().astype("float32")
            logits_l2 = np.clip(logits_l2, 0.0, 1.0)
            logits = logits*0.5 + logits_l2*0.5
        true_ans_list.append(targets)
        preds_cat.append(logits)

    all_true_ans = np.concatenate(true_ans_list, axis=0)
    all_preds = np.concatenate(preds_cat, axis=0)

    return all_preds, all_true_ans, val_loss / (step + 1)


if 1:
    delta = 0.5
    best_score = -999
    best_epoch = 0
    for epoch in range(1, epochs + 1):

        print("Starting {} epoch...".format(epoch))
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print('Mean train loss: {}'.format(round(tr_loss, 5)))

        val_pred, y_true, val_loss = validate(model, val_loader, criterion, device)
        score = accuracy_score(y_true, (val_pred > delta).astype(int))
        print('Mean valid loss: {} score: {}'.format(round(val_loss, 5), round(score, 5)))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            # torch.save(model.state_dict(), os.path.join(OUT_DIR, '{}_fold{}.pth'.format(EXP_ID, fold_id)))
            # np.save(os.path.join(OUT_DIR, "{}_fold{}.npy".format(EXP_ID, fold_id)), val_pred)
        scheduler.step()

    print("best score={} on epoch={}".format(best_score, best_epoch))


