import spacy
import json
import re
from copy import deepcopy
from tqdm import tqdm
spacy_processor = spacy.load("en_core_web_sm")


def get_first_verb(input_text):
    continuous_predicate = []
    past_state_verb = False
    for each_word in input_text.split():
        is_verb = False
        is_prep = False
        doc = spacy_processor(each_word)
        for each_token in doc:
            if each_token.pos_ in ["AUX", "VERB"]:
                is_verb = True
            elif each_token.pos_ == "ADP":
                is_prep = True
        if is_verb and (past_state_verb is False):
            continuous_predicate.append([each_word])
            past_state_verb = True
        elif (is_verb or is_prep) and past_state_verb:
            continuous_predicate[-1].append(each_word)
        elif continuous_predicate:
            return " ".join(continuous_predicate[0])
    return None


def get_first_clause(input_text):
    doc = spacy_processor(input_text)
    for token in doc:
        # Check for the start of a clause
        if token.dep_ in ['advcl', 'ccomp', 'xcomp', 'acl', 'relcl', 'prep']:
            # Find the start and end index of the clause
            start_index = token.left_edge.idx
            end_index = token.right_edge.idx + len(token.right_edge)
            # Return the exact substring from the original input text
            return input_text[start_index:end_index]
    return None


output_list = []
input_file = '../../data/build/lingnli/train_disc_merged.jsonl'
output_file = '../../data/build/lingnli/train_disc_merged_delim1.jsonl'

with open(input_file, 'r', encoding='utf-8-sig') as file:
    line_id = 0
    for line in tqdm(file):
        data = json.loads(line)
        # output_list.append(data)
        # delimiter = ","
        # tmp_d = deepcopy(data)
        # tmp_pred = get_first_verb(tmp_d["hypothesis"])
        # if tmp_pred:
        #     tmp_d["hypothesis"] = tmp_d["hypothesis"].replace(" " + tmp_pred + " ", ", " + tmp_pred + ", ")
        # tmp_clause = get_first_clause(tmp_d["hypothesis"])
        # if tmp_clause:
        #     tmp_d["hypothesis"] = tmp_d["hypothesis"].replace(tmp_clause, ", " + tmp_clause)
        # escaped_delimiter = re.escape(delimiter)
        # new_text = tmp_d["hypothesis"]
        # while True:
        #     old_text = new_text
        #     new_text = re.sub(r'\s+', ' ', new_text).strip(" ," + delimiter)
        #     new_text = new_text.replace(delimiter + " " + delimiter, delimiter)
        #     new_text = new_text.replace(", " + delimiter, delimiter)
        #     new_text = new_text.replace(delimiter + " ,", delimiter)
        #     if new_text == old_text:
        #         break
        # tmp_d["hypothesis"] = new_text
        # output_list.append(tmp_d)

        delimiter = "|"
        tmp_d = deepcopy(data)
        tmp_pred = get_first_verb(tmp_d["hypothesis"])
        if tmp_pred:
            tmp_d["hypothesis"] = tmp_d["hypothesis"].replace(" " + tmp_pred + " ", " | " + tmp_pred + " | ")
        tmp_clause = get_first_clause(tmp_d["hypothesis"])
        if tmp_clause:
            tmp_d["hypothesis"] = tmp_d["hypothesis"].replace(tmp_clause, "| " + tmp_clause)
        escaped_delimiter = re.escape(delimiter)
        new_text = tmp_d["hypothesis"]
        while True:
            old_text = new_text
            new_text = re.sub(r'\s+', ' ', new_text).strip(" ," + delimiter)
            new_text = new_text.replace(delimiter + " " + delimiter, delimiter)
            new_text = new_text.replace(", " + delimiter, delimiter)
            new_text = new_text.replace(delimiter + " ,", delimiter)
            if new_text == old_text:
                break
        tmp_d["hypothesis"] = new_text
        output_list.append(tmp_d)

with open(output_file, encoding='utf-8-sig', mode='w') as out_f:
    for each_line in output_list:
        out_f.write(json.dumps(each_line) + '\n')
