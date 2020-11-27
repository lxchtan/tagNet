# coding=utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm

from torch.nn import CrossEntropyLoss
from collections import OrderedDict

from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel, BertPredictionHeadTransform
from pytorch_pretrained_bert.tokenization import BertTokenizer

from med import levenshtein_revise

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

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

    act_tag = self.transform_act(sequence_output.detach())
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
      self._read_json(os.path.join(data_dir, "test_unlabelled.json")), "dev")

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
        labels = None
        if 'semantic' in dic:
          labels = dic['semantic']
        results.append((utterance, labels))
    return results

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

  datas = json.load(open(os.path.join(data_dir, "test_unlabelled.json")), object_pairs_hook=OrderedDict)
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
                      # default="../data/weather",
                      type=str,
                      required=True,
                      help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
  parser.add_argument("--bert_model", default="bert-base-chinese", type=str,
                      # required=True,
                      help="Bert pre-trained model selected in the list: bert-base-uncased, "
                           "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                           "bert-base-multilingual-cased, bert-base-chinese.")
  parser.add_argument("--task_name",
                      default="map",
                      type=str,
                      # required=True,
                      help="The name of the task to train.")
  parser.add_argument("--output_dir",
                      # default="../output/all_weather",
                      type=str,
                      required=True,
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
                      default=True,
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
                      default=3.0,
                      type=float,
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
  args = parser.parse_args()

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
  _ontology_file_path = os.path.join(data_dir, "ontology.json")
  get_result(tokenizer, preds, _tag, _act_tag, _request, _inform, _act, data_dir, output_file)

  levenshtein_revise(input=output_file , output=output_file, ontology_file_path=_ontology_file_path)

if __name__ == "__main__":
  main()
