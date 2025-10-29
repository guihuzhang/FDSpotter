import codecs
import json
import os
import re
import sys
import copy
import logging
from os.path import exists

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))


def load_model_and_tokenizer(model_name="https://huggingface.co/Inria-CEDAR/FDSpotter-DeBERTa-V3-Large", device='cuda'):
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if device == 'cuda':
        model.cuda()
    elif device != 'cuda':
        model.cpu()
    return tokenizer, model


def clean_pipe_dash_space(input_string):
    previous_string = ""
    while previous_string != input_string:
        previous_string = input_string
        input_string = re.sub(r'\|\s*([-\s–—\u2010\u2212\u2012\u2015\u00AD\u2011]+)\s*\|', '|', input_string)
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    return cleaned_string


def evaluate_extractions_segments_batch(ext_str_list, txt_chunks, tokenizer, model, batch_size=32):
    all_pairs = []
    for r_idx, relation_str in enumerate(ext_str_list):
        for chunk in txt_chunks:
            all_pairs.append((chunk, relation_str, r_idx))
    # result_list = [0.0]*len(all_pairs)
    result_list = []
    total = len(all_pairs)
    for start in range(0, total, batch_size):
        batch = all_pairs[start: start + batch_size]
        pairs_for_inference = [(item[0], item[1]) for item in batch]
        ent_scores = sentence_cls_score(pairs_for_inference, model, tokenizer)[:, 0].tolist()
        for i, score in enumerate(ent_scores):
            result_list.append(score)
    # take the max of each chunk
    final_scores = [0.0] * len(ext_str_list)
    for (chunk, relation_str, r_idx), sc in zip(all_pairs, result_list):
        if sc > final_scores[r_idx]:
            final_scores[r_idx] = sc
    return final_scores


def select_best_ext(gpt_extractions, summary_chunks, model, tokenizer):
    """
    gpt_extractions: [ { "atomic facts": [...], "discourse relations": [...] }, ... ]
    Return: list of (extraction_obj, avg_score)
    """
    results = []
    for each_ext in gpt_extractions:
        all_relations = []
        atomic_facts = each_ext.get("atomic facts", [])
        disc_rels = each_ext.get("discourse relations", [])
        atomic_facts = [{"relation": clean_pipe_dash_space(af.strip(" \u002D\u2013\u2014\u2010\u2212\u2012\u2015\u00AD\u2011|"))}
                        for af in atomic_facts]
        disc_rels = [{"relation": clean_pipe_dash_space(dr.strip(" \u002D\u2013\u2014\u2010\u2212\u2012\u2015\u00AD\u2011|"))}
                     for dr in disc_rels]
        for atom in atomic_facts:
            all_relations.append(atom["relation"])
        for disc in disc_rels:
            all_relations.append(disc["relation"])
        if not all_relations:
            results.append(({'atomic facts': atomic_facts, "discourse relations": disc_rels}, 0.0))
            continue
        max_for_each_relation = evaluate_extractions_segments_batch(
            all_relations, summary_chunks, tokenizer, model, )
        avg_score = float(np.mean(max_for_each_relation))
        idx_atom = 0
        for atom in atomic_facts:
            atom["cls conf"] = max_for_each_relation[idx_atom]
            idx_atom += 1
        for i, disc in enumerate(disc_rels):
            disc["cls conf"] = max_for_each_relation[idx_atom]
            idx_atom += 1
        results.append(({'atomic facts': atomic_facts, "discourse relations": disc_rels}, avg_score))
    return results


def sentence_cls_score(input_strings, predicate_cls_model, predicate_cls_tokenizer):
    with torch.no_grad():
        tokenized_cls_input = predicate_cls_tokenizer(input_strings, truncation="longest_first", padding=True, max_length=1280, return_token_type_ids=True)
        input_id_cls = torch.Tensor(tokenized_cls_input['input_ids']).long().to(torch.device("cuda"))
        token_type_ids = torch.Tensor(tokenized_cls_input['token_type_ids']).long().to(torch.device("cuda"))
        attention_mask_cls = torch.Tensor(tokenized_cls_input['attention_mask']).long().to(torch.device("cuda"))
        prev_cls_output = predicate_cls_model(input_id_cls, attention_mask=attention_mask_cls, token_type_ids=token_type_ids)
        softmax_cls_output = torch.softmax(prev_cls_output.logits, dim=1, )
        return softmax_cls_output


def parse_gen_ref(refs_path, hyps_path, num_refs):
    logging.info('STARTING TO PARSE INPUTS...')
    print('STARTING TO PARSE INPUTS...')
    # references
    references = []
    for i in range(num_refs):
        fname = refs_path + str(i) if num_refs > 1 else refs_path
        with codecs.open(fname, 'r', 'utf-8') as f:
            texts = f.readlines()
            for j, text in enumerate(texts):
                if len(references) <= j:
                    references.append([text.strip("\n")])
                else:
                    references[j].append(text.strip("\n"))
    # hypothesis
    with codecs.open(hyps_path, 'r', 'utf-8') as f:
        hypothesis = f.read().split('\n')

    logging.info('FINISHING TO PARSE INPUTS...')
    print('FINISHING TO PARSE INPUTS...')
    return references, hypothesis


def main():
    parameters = json.load(open(sys.argv[1]))
    device = 'cuda'
    tokenizer, model = load_model_and_tokenizer(model_name=parameters["model-base"], device=device)

    ext_dict = {}
    with open(parameters["gen-extraction"], 'r', encoding='utf-8-sig') as f:
        tmp_all_ext = json.load(f)
        for each_entry in tqdm(tmp_all_ext):
            tmp_txt = each_entry["txt"].strip()
            tmp_extractions = [{"atomic facts": x["atomic facts"],
                                "discourse relations": x["discourse relations"]}
                               for x in each_entry["ext"]]
            processed_ext = []
            for each_ext in tmp_extractions:
                tmp_atom = each_ext["atomic facts"]
                tmp_disc = each_ext["discourse relations"]
                processed_ext.append({"atomic facts": tmp_atom, "discourse relations": tmp_disc})
            if tmp_txt not in ext_dict:
                ext_dict[tmp_txt] = processed_ext
            else:
                ext_dict[tmp_txt].extend(processed_ext)

    with open(parameters["gt-extraction"], 'r', encoding='utf-8-sig') as f:
        tmp_all_ext = json.load(f)
        for each_entry in tqdm(tmp_all_ext):
            tmp_txt = each_entry["txt"].strip()
            tmp_extractions = [{"atomic facts": x["atomic facts"],
                                "discourse relations": x["discourse relations"]}
                               for x in each_entry["ext"]]
            processed_ext = []
            for each_ext in tmp_extractions:
                tmp_atom = each_ext["atomic facts"]
                tmp_disc = each_ext["discourse relations"]
                processed_ext.append({"atomic facts": tmp_atom, "discourse relations": tmp_disc})
            if tmp_txt not in ext_dict:
                ext_dict[tmp_txt] = processed_ext
            else:
                ext_dict[tmp_txt].extend(processed_ext)

    for each_entry in tqdm(ext_dict):
        tmp_ext = select_best_ext(ext_dict[each_entry], [each_entry], model, tokenizer)
        ext_dict[each_entry] = sorted(tmp_ext, key=lambda x: x[-1], reverse=True)

    team_scores = {}
    files = [f.path for f in os.scandir(parameters['teams-samples'])]
    for file_name in files:
        team_name = file_name.split('/')[-1]
        if team_name not in team_scores:
            team_scores[team_name] = {}

        references, hypothesis = parse_gen_ref(parameters['references'], file_name, parameters['num_references'])
        for idx in tqdm(range(0, len(references))):
            sum_scores_hyp = 0
            len_hyp = 0
            sum_scores_ref = 0
            len_ref = 0

            tmp_hypothesis = hypothesis[idx]
            tmp_references = references[idx]
            if tmp_hypothesis not in ["", "null"]:
                hyp_ext = ext_dict[tmp_hypothesis][0]
                ref_ext = [ext_dict[x][0] for x in tmp_references if x in ext_dict]
                # print("hyp", tmp_hypothesis)
                # print(hyp_ext)
                # print("ref", tmp_references)
                # print(ref_ext)
                valid_ext_hyp = []
                for each_atom in hyp_ext[0]["atomic facts"]:
                    if each_atom["cls conf"] > .5:
                        valid_ext_hyp.append(each_atom["relation"])
                for each_disc in hyp_ext[0]["discourse relations"]:
                    if each_disc["cls conf"] > .5:
                        valid_ext_hyp.append(each_disc["relation"])
                hyp_to_ref_scores = evaluate_extractions_segments_batch(valid_ext_hyp, tmp_references, tokenizer, model, 32)
                # print(hyp_to_ref_scores)
                sum_scores_hyp += np.sum(hyp_to_ref_scores)
                len_hyp += len(hyp_to_ref_scores)
                # print(sum_scores_hyp)
                # print(len_hyp)
                valid_ext_ref = []
                ref_to_hyp_scores = []
                tmp_sum = 0.
                tmp_len = 0
                for each_ext in ref_ext:
                    valid_ext_ref.append([])
                    for each_atom in each_ext[0]["atomic facts"]:
                        if each_atom["cls conf"] > .5:
                            valid_ext_ref[-1].append(each_atom["relation"])
                    for each_disc in each_ext[0]["discourse relations"]:
                        if each_disc["cls conf"] > 0.5:
                            valid_ext_ref[-1].append(each_disc["relation"])
                    ref_to_hyp_scores.append(evaluate_extractions_segments_batch(valid_ext_ref[-1], [tmp_hypothesis], tokenizer, model, 32))
                    tmp_sum += np.sum(ref_to_hyp_scores[-1])
                    tmp_len += len(ref_to_hyp_scores[-1])
                sum_scores_ref += tmp_sum / len(ref_to_hyp_scores)
                len_ref += tmp_len / len(ref_to_hyp_scores)
                # print(ref_to_hyp_scores)
                # print(sum_scores_ref)
                # print(len_ref)
            # print(sum_scores_ref, len_ref)
            # print(sum_scores_hyp, len_hyp)
            final_remark = (sum_scores_ref + sum_scores_hyp) / (len_hyp + len_ref) if (len_hyp > 0 and len_ref > 0) else 0.
            team_scores[team_name][idx] = final_remark
        # print(team_scores[team_name])
    folder_path = parameters["metrics-path"] + "/fact-overlap-large523"
    if not exists(folder_path):
        os.mkdir(folder_path)
    for each_team in team_scores:
        wb = open(folder_path + "/" + each_team, "w")
        for line_id in range(0, parameters["no-samples"]):
            tmp_key = line_id
            if tmp_key not in team_scores[each_team]:
                wb.write("0\n")
            else:
                wb.write(str(team_scores[each_team][tmp_key]) + "\n")
        wb.close()
    print("finished writing to files!")


if __name__ == '__main__':
    main()
