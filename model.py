# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss
from collections import OrderedDict

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, BertPredictionHeadTransform
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import time
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    filename='./experiment/bert/log_{}.txt'.format(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())),
                    filemode='w',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForCatslu(BertPreTrainedModel):
  def __init__(self, config, num_labels, num_slot_tags, num_acts):
    super(BertForCatslu, self).__init__(config)

    self.num_labels = num_labels
    self.num_slot_tags = num_slot_tags
    self.num_acts = num_acts

    self.bert = BertModel(config)

    self.transform = BertPredictionHeadTransform(config)
    self.slot_tags_linear = torch.nn.Linear(config.hidden_size, num_slot_tags)

    self.transform_act = BertPredictionHeadTransform(config)
    self.act_linear = torch.nn.Linear(config.hidden_size, num_acts)

    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    self.apply(self.init_bert_weights)

  def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, tags=None, acts=None):
    sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    slot_tag = self.transform(sequence_output)
    slot_tag_logits = self.slot_tags_linear(slot_tag)

    act_tag = self.transform_act(sequence_output)
    act_tag_logits = self.act_linear(act_tag)

    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      loss_fct2 = CrossEntropyLoss(ignore_index=-1)
      loss2 = loss_fct2(slot_tag_logits.view(-1, self.num_slot_tags), tags.view(-1))
      loss3 = loss_fct2(act_tag_logits.view(-1, self.num_acts), acts.view(-1))
      return loss + loss2 + loss3
    else:
      return logits, slot_tag_logits, act_tag_logits


class InputExample(object):
  def __init__(self, guid, text_a, text_b=None, label=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class InputFeatures(object):
  def __init__(self, input_ids, input_mask, segment_ids, label_id, slot_ids, act_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.slot_ids = slot_ids
    self.act_ids = act_ids


class CatsluProcessor():
  def get_train_examples(self, data_dir):
    """See base class."""
    logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.json")))
    return self._create_examples(
      self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self._read_json(os.path.join(data_dir, "development.json")), "dev")

  def get_slots(self, data_dir):
    datas = json.loads(open(os.path.join(data_dir, "ontology.json")).read(), object_pairs_hook=OrderedDict)
    _request = datas["slots"]["requestable"] + ["None"]
    _inform = list(datas["slots"]["informable"].keys())
    _act = [k for k, v in datas["acts"].items() if v==0]
    return _request, _inform, _act

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = line[0]
      label = line[1]
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  @classmethod
  def _read_json(cls, file_name, utt_mode='asr'):
    datas = json.loads(open(file_name).read(), object_pairs_hook=OrderedDict)
    results = []
    for data in datas:
      utterances = data['utterances']
      for dic in utterances:
        if utt_mode == 'manual':
          utterance = dic['manual_transcript']
        else:
          utterance = dic['asr_1best']
        labels = dic.get('semantic')
        results.append((utterance, labels))
    return results

def convert_examples_to_features(examples, _request, _inform, _act, max_seq_length,
                                 tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  label_map = {label: i for i, label in enumerate(_request)}
  acts_map = {act: i for i, act in enumerate(_act)}
  acts_map.update({"O": -1})
  slot_tags_map = {"O": 0}
  for i, slot in enumerate(_inform, 1):
    slot_tags_map["B-{}".format(slot)] = i * 2 - 1
    slot_tags_map["I-{}".format(slot)] = i * 2

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 1000 == 0:
      logger.info("Writing example %d of %d" % (ex_index, len(examples)))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    label = "None"
    slot_tag = ["O" for text in tokens_a]
    act_tag = ["O" for text in tokens_a]

    if len(example.label) != 0:
      for it in example.label:
        if it[0] == 'request':
          label = it[1]
        else:
          tokens_value = tokenizer.tokenize(it[2])
          start = -1
          for i in range(len(tokens_a) - len(tokens_value) + 1):
            if tokens_value == tokens_a[i: i+len(tokens_value)]:
              start = i
              break
          if start != -1:
            slot_tag[start] = "B-{}".format(it[1])
            act_tag[start] = it[0]
            if len(tokens_value) > 1:
              slot_tag[start+1:start+len(tokens_value)] = ["I-{}".format(it[1])] * (len(tokens_value) - 1)

    label_id = label_map[label]

    slot_ids = [slot_tags_map[slot] for slot in slot_tag]
    act_ids = [acts_map[act] for act in act_tag]

    if tokens_b:
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(slot_ids) == len(input_ids) - 2

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    slot_ids = [-1] + slot_ids + [-1] * (len(padding) + 1)
    act_ids = [-1] + act_ids + [-1] * (len(padding) + 1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(slot_ids) == max_seq_length
    assert len(act_ids) == max_seq_length

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("tokens: %s" % " ".join(
        [str(x) for x in tokens]))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("slot_ids: %s" % " ".join([str(x) for x in slot_ids]))
      logger.info("act_ids: %s" % " ".join([str(x) for x in act_ids]))
      logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logger.info(
        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label_id))



    features.append(
      InputFeatures(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                    slot_ids=slot_ids,
                    act_ids=act_ids))
  return features

def convert_test_examples_to_features(examples, max_seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 1000 == 0:
      logger.info("Writing example %d of %d" % (ex_index, len(examples)))

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
      tokens += tokens_b + ["[SEP]"]
      segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("tokens: %s" % " ".join(
        [str(x) for x in tokens]))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      logger.info(
        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    features.append(
      InputFeatures(input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=None,
                    slot_ids=None,
                    act_ids=None))
  return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def get_result(tokenizer, preds, _tag, _act_tag, _request, _inform, _act, data_dir, output_file, utt_mode='asr'):
  label_map_inv = {i: label for i, label in enumerate(_request)}
  acts_map_inv = {i: act for i, act in enumerate(_act)}
  slot_tags_map_inv = {0: "O"}
  for i, slot in enumerate(_inform, 1):
    slot_tags_map_inv[i * 2 - 1] = "B-{}".format(slot)
    slot_tags_map_inv[i * 2] = "I-{}".format(slot)

  datas = json.load(open(os.path.join(data_dir, "development.json")), object_pairs_hook=OrderedDict)
  _index = 0
  for data in datas:
    for anno_utterance in data['utterances']:
      text = anno_utterance['asr_1best'] if utt_mode == 'asr' else anno_utterance['manual_transcript']
      sentence_tag = _tag[_index]
      sentence_act = _act_tag[_index]
      token_a = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
      len_token = len(token_a)
      slots = [slot_tags_map_inv[i] for i in sentence_tag[1:len_token + 1]]
      acts = [acts_map_inv[i] for i in sentence_act[1:len_token + 1]]
      semantic = []
      _i = 0
      while _i < len_token:
        if slots[_i].startswith("B-"):
          now_slot = slots[_i][2:]
          end_pos = _i + 1
          stop_flag = True
          for _j in range(_i + 1, len_token):
            end_pos = _j
            if slots[_j] != "I-{}".format(now_slot):
              stop_flag = False
              break
          if stop_flag:
            end_pos = len_token
          semantic.append([acts[_i], now_slot, ''.join(tokenizer.convert_ids_to_tokens(token_a[_i:end_pos]))])
          _i = end_pos
        else:
          _i += 1
      if preds[_index] != len(_request) - 1:
        semantic.append(["request", label_map_inv[preds[_index]]])
      anno_utterance['semantic'] = semantic
      _index += 1

  json.dump(datas, open(output_file, 'w', encoding='utf-8'), ensure_ascii=False)


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir",
                      default="./data/weather",
                      type=str,
                      # required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--bert_model", default="bert-base-chinese", type=str,
                      # required=True,
                      help="Bert pre-trained model selected in the list: bert-base-uncased, "
                           "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                           "bert-base-multilingual-cased, bert-base-chinese.")
  parser.add_argument("--task_name",
                      default="weather",
                      type=str,
                      # required=True,
                      help="The name of the task to train.")
  parser.add_argument("--output_dir",
                      default="./experiment/bert/weather",
                      type=str,
                      # required=True,
                      help="The output directory where the model predictions and checkpoints will be written.")

  ## Other parameters
  parser.add_argument("--cache_dir",
                      default="",
                      type=str,
                      help="Where do you want to store the pre-trained models downloaded from s3")
  parser.add_argument("--max_seq_length",
                      default=64,
                      type=int,
                      help="The maximum total input sequence length after WordPiece tokenization. \n"
                           "Sequences longer than this will be truncated, and sequences shorter \n"
                           "than this will be padded.")
  parser.add_argument("--do_train",
                      default=False,
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval",
                      default=False,
                      action='store_true',
                      help="Whether to run eval on the dev set.")
  parser.add_argument("--do_lower_case",
                      default=True,
                      action='store_true',
                      help="Set this flag if you are using an uncased model.")
  parser.add_argument("--train_batch_size",
                      default=32,
                      type=int,
                      help="Total batch size for training.")
  parser.add_argument("--eval_batch_size",
                      default=8,
                      type=int,
                      help="Total batch size for eval.")
  parser.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate for Adam.")
  parser.add_argument("--num_train_epochs",
                      default=10,
                      type=int,
                      help="Total number of training epochs to perform.")
  parser.add_argument("--warmup_proportion",
                      default=0.1,
                      type=float,
                      help="Proportion of training to perform linear learning rate warmup for. "
                           "E.g., 0.1 = 10%% of training.")
  parser.add_argument("--no_cuda",
                      action='store_true',
                      help="Whether not to use CUDA when available")
  parser.add_argument("--local_rank",
                      type=int,
                      default=-1,
                      help="local_rank for distributed training on gpus")
  parser.add_argument('--seed',
                      type=int,
                      default=42,
                      help="random seed for initialization")
  parser.add_argument('--gradient_accumulation_steps',
                      type=int,
                      default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass.")
  parser.add_argument('--fp16',
                      action='store_true',
                      help="Whether to use 16-bit float precision instead of 32-bit")
  parser.add_argument('--loss_scale',
                      type=float, default=0,
                      help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                           "0 (default value): dynamic loss scaling.\n"
                           "Positive power of 2: static loss scaling value.\n")
  parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
  parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
  args = parser.parse_args()

  if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
  else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')

  logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

  logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args.local_rank != -1), args.fp16))

  if args.gradient_accumulation_steps < 1:
    raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
      args.gradient_accumulation_steps))

  args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)

  if not args.do_train and not args.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  processor = CatsluProcessor()

  _request, _inform, _act = processor.get_slots(args.data_dir)
  num_labels = len(_request)
  num_slot_tags = len(_inform)*2 + 1
  num_acts = len(_act)


  train_examples = None
  num_train_optimization_steps = None
  if args.do_train:
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_optimization_steps = int(
      len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    if args.local_rank != -1:
      num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

  # Prepare model
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForCatslu.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels, num_slot_tags=num_slot_tags, num_acts=num_acts)

    if args.fp16:
      model.half()
    model.to(device)
    if args.local_rank != -1:
      try:
        from apex.parallel import DistributedDataParallel as DDP
      except ImportError:
        raise ImportError(
          "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

      model = DDP(model)
    elif n_gpu > 1:
      model = torch.nn.DataParallel(model)

  # Prepare optimizer
  if args.do_train:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
      try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
      except ImportError:
        raise ImportError(
          "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

      optimizer = FusedAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            bias_correction=False,
                            max_grad_norm=1.0)
      if args.loss_scale == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
      else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
      warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                           t_total=num_train_optimization_steps)

    else:
      optimizer = BertAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           t_total=num_train_optimization_steps)

  global_step = 0

  if args.do_train:
    train_features = convert_examples_to_features(
      train_examples, _request, _inform, _act, args.max_seq_length, tokenizer)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_slot_ids = torch.tensor([f.slot_ids for f in train_features], dtype=torch.long)
    all_act_ids = torch.tensor([f.act_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_slot_ids, all_act_ids)
    if args.local_rank == -1:
      train_sampler = RandomSampler(train_data)
    else:
      train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    pbar = trange(int(args.num_train_epochs), desc="Epoch")
    for e in pbar:
      tr_loss = 0
      nb_tr_examples, nb_tr_steps = 0, 0
      for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, slot_ids, act_ids = batch

        loss = model(input_ids, segment_ids, input_mask, labels=label_ids, tags=slot_ids, acts=act_ids)

        if n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        if args.fp16:
          optimizer.backward(loss)
        else:
          loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        pbar.set_postfix(loss=loss.item())
        logging.info("epoch = {}, step = {}, loss = {:.2f}".format(e, step, loss.item()))
        if (step + 1) % args.gradient_accumulation_steps == 0:
          if args.fp16:
            # modify learning rate with special warm up BERT uses
            # if args.fp16 is False, BertAdam is used that handles this automatically
            lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr_this_step
          optimizer.step()
          optimizer.zero_grad()
          global_step += 1
    pbar.close()

  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

  if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    model = BertForCatslu.from_pretrained(args.output_dir, num_labels=num_labels, num_slot_tags=num_slot_tags, num_acts=num_acts)
    tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    model.to(device)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_test_examples_to_features(eval_examples, args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    preds = []

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader, desc="Evaluating"):
      input_ids = input_ids.to(device)
      input_mask = input_mask.to(device)
      segment_ids = segment_ids.to(device)

      with torch.no_grad():
        logits, slot_tag_logits, act_logits = model(input_ids, segment_ids, input_mask, labels=None, tags=None)

      if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
        preds.append(slot_tag_logits.detach().cpu().numpy())
        preds.append(act_logits.detach().cpu().numpy())
      else:
        preds[0] = np.append(
          preds[0], logits.detach().cpu().numpy(), axis=0)
        preds[1] = np.append(
          preds[1], slot_tag_logits.detach().cpu().numpy(), axis=0)
        preds[2] = np.append(
          preds[2], act_logits.detach().cpu().numpy(), axis=0)

    preds, _tag, _act_tag = preds

    preds = np.argmax(preds, axis=1)
    _tag = np.argmax(_tag, axis=-1)
    _act_tag = np.argmax(_act_tag, axis=-1)

    data_dir = args.data_dir
    output_file = os.path.join(args.output_dir, "write_results.json")
    get_result(tokenizer, preds, _tag, _act_tag, _request, _inform, _act, data_dir, output_file)

if __name__ == "__main__":
  main()

