#!/usr/bin/env python3

'''
@Time   : 2019-04-04 13:33:23
@Author : su.zhu
@Desc   : 
'''

import os
import json

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
