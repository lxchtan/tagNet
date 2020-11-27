import Levenshtein
import os
import json
import argparse
from copy import deepcopy

def load_ontology(ontology_file_path):
    src_base_dir = os.path.dirname(ontology_file_path)
    ontology = json.load(open(ontology_file_path))

    for slot in list(ontology['slots']['informable']):
        values = ontology['slots']['informable'][slot]
        ## load lexicon file
        if type(values) == str:
            values = [line.strip() for line in open(os.path.join(src_base_dir, values)) if line.strip() != ""]
        ontology['slots']['informable'][slot] = set(values)

    return ontology

def get_min_levenshtein(s, slist):
  min_dis = 128
  min_sent = s
  for t in slist:
    dis = Levenshtein.distance(s, t)
    if dis < min_dis:
      min_dis = dis
      min_sent = t
  if min_dis >= len(s)-1:
    min_sent = None
  return min_sent

def levenshtein_revise(input, output, ontology_file_path):
  ontolopy = load_ontology(ontology_file_path)

  with open(input) as in_file:
    slu_pred_data = json.load(in_file)

  for pred_dialogue in slu_pred_data:
    for pred_utterance in pred_dialogue['utterances']:
      pred_semantics = pred_utterance["semantic"]

      # add whole sentence
      if pred_semantics == [] and len(pred_utterance['asr_1best']) > 2:
        for k, v in ontolopy['slots']['informable'].items():
          if pred_utterance['asr_1best'] in v:
            pred_semantics.append(['inform', k, pred_utterance['asr_1best']])
            break

      for ps in pred_semantics:
        if ontolopy['acts'][ps[0]] == 0:
          if ps[2] not in ontolopy['slots']['informable'][ps[1]]:
            flag = True
            for k, v in ontolopy['slots']['informable'].items():
              if ps[2] in v:
                ps[1] = k
                flag = False
                break
            if flag:
              ps[2] = get_min_levenshtein(ps[2], ontolopy['slots']['informable'][ps[1]])

  for pred_dialogue in slu_pred_data:
    for pred_utterance in pred_dialogue['utterances']:
      pred_semantics = pred_utterance["semantic"]
      for ps in deepcopy(pred_semantics):
        if ontolopy['acts'][ps[0]] == 0 and ps[2] == None:
          pred_semantics.remove(ps)

  json.dump(slu_pred_data, open(output, 'w', encoding='utf-8'), ensure_ascii=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_dir", type=str, required=True, help='For example: experiment/bert/music')
  parser.add_argument("--ontology_path", help='For example: data/music/ontology.json', required=True)
  parser.add_argument("--input_file", default="write_results.json")
  parser.add_argument("--output_file", default="write_results_med.json")
  args = parser.parse_args()

  _input = os.path.join(args.output_dir, args.input_file)
  _output = os.path.join(args.output_dir, args.output_file)

  levenshtein_revise(input=_input, output=_output, ontology_file_path = args.ontology_path)