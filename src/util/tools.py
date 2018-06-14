#!/usr/bin/env python3
# coding: utf8


from ..conf.config import Config
from ..preprocessing.load import load_dict

import logging
import re

def id_to_term(accession_id):
    '''getting disease name from id'''
    accession_id=accession_id.replace('+','|')
    if '|' not in accession_id:
        return _get_term(accession_id)
    else:
        l=[]
        for id in accession_id.split('|'):
            l.append(_get_term(id))
        flat_list = [item for sublist in l for item in sublist]
        return flat_list

def _get_term(accession_id):
    #seems to be awfully slow
    conf=Config()
    terminology=load_dict(conf)
    try:
        if accession_id == 'NIL':
            return ['NIL']
        entries = terminology._by_id[accession_id]
        return [e.name for e in entries]
    except Exception as e:
        logging.exception(e)

def add_disease_name(filename,correct = False, reachable = None):
    '''add disease name into the saved results
       optional arguments:
           correct: True or False or None. Default is False. If True,
           also prints the disease names for correct predictions
           reachable: True, False, or None. Default is None. If set
           True or False, prints only the predictions with given value
    '''

    newname = filename.replace('.txt','_disease_added.txt')
    output_file=open(newname,'w',encoding='ascii')
    output_file.write('DOC_ID\tSTART\tEND\tMENTION\tREF_ID\tREF_DISEASE\tPRED_ID\tPRED_DISEASE\tCORRECT\tN_IDS\tREACHABLE\n')

    with open(filename,'r',encoding='ascii') as file:
        #header row
        next(file)
        for line in file:
            _, _, _, _, REF_ID, PRED_ID, CORRECT, _, REACHABLE = line.split('\t')
            CORRECT = (CORRECT == 'True')
            REACHABLE = (REACHABLE == 'True')
            if (correct == CORRECT or correct == None) and (reachable == REACHABLE or reachable == None):
                output_file.write(line)
                output_file.write(_print_entry(REF_ID, PRED_ID))

    output_file.close()

def _print_entry(REF_ID, PRED_ID):
    REF_DISEASE = id_to_term(REF_ID)
    PRED_DISEASE = id_to_term(PRED_ID)
    DISEASES = str(REF_DISEASE)+'\t'+str(PRED_DISEASE)+'\n'
    return DISEASES
