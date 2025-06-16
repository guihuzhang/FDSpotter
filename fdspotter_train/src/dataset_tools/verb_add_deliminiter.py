import spacy
import json
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
        # Check for starting of a clause
        if token.dep_ in ['advcl', 'ccomp', 'xcomp']:
            return ' '.join([tok.text for tok in token.subtree])
    return None


output_list = []
with open('../../data/build/snli/train.jsonl', 'r', encoding='utf-8-sig') as file:
    line_id = 0
    for line in tqdm(file):
        data = json.loads(line)
        if line_id % 3 == 0:
            tmp_pred = get_first_verb(data["hypothesis"])
            if tmp_pred:
                data["hypothesis"] = data["hypothesis"].replace(" " + tmp_pred + " ", ", " + tmp_pred + ", ")
        elif line_id % 3 == 1:
            tmp_pred = get_first_verb(data["hypothesis"])
            if tmp_pred:
                data["hypothesis"] = data["hypothesis"].replace(" " + tmp_pred + " ", " | " + tmp_pred + " | ")
        line_id += 1
        output_list.append(data)

with open('../../data/build/snli/train_delim.jsonl', encoding='utf-8-sig', mode='w') as out_f:
    for each_line in output_list:
        out_f.write(json.dumps(each_line) + '\n')
