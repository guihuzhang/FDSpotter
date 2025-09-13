import csv
import json
import re
import neuralcoref
import numpy as np
import spacy
import torch

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from mosestokenizer import MosesDetokenizer
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)


def load_model_and_tokenizer(model_name="Inria-CEDAR/FDSpotter-DeBERTa-V3-Large", device='cuda'):
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    if device == 'cuda':
        model.cuda()
    elif device != 'cuda':  # Explicitly place the model on CPU if not using CUDA
        model.cpu()
    return tokenizer, model


def sentence_cls_score(input_strings, predicate_cls_model, predicate_cls_tokenizer):
    with torch.no_grad():
        tokenized_cls_input = predicate_cls_tokenizer(input_strings, truncation="longest_first", padding=True, max_length=1280, return_token_type_ids=True)
        input_id_cls = torch.Tensor(tokenized_cls_input['input_ids']).long().to(torch.device("cuda"))
        token_type_ids = torch.Tensor(tokenized_cls_input['token_type_ids']).long().to(torch.device("cuda"))
        attention_mask_cls = torch.Tensor(tokenized_cls_input['attention_mask']).long().to(torch.device("cuda"))
        prev_cls_output = predicate_cls_model(input_id_cls, attention_mask=attention_mask_cls,
                                              token_type_ids=token_type_ids)
        softmax_cls_output = torch.softmax(prev_cls_output.logits, dim=1, )
        return softmax_cls_output


def assemble_gpt_conf(ext_lines):
    ext_and_conf = []
    for each_line in ext_lines:
        tmp_ext = clean_pipe_dash_space(each_line[0].strip(" \u002D\u2013\u2014\u2010\u2212\u2012\u2015\u00AD\u2011|"))
        tmp_conf = []
        for cw in each_line[1]:
            if cw[1].strip(" \u002D\u2013\u2014\u2010\u2212\u2012\u2015\u00AD\u2011|\n"):
                tmp_conf.append(cw[0])
        ext_and_conf.append({"relation": tmp_ext, "gpt conf": sum(tmp_conf) / len(tmp_conf)})
    return ext_and_conf


def clean_pipe_dash_space(input_string):
    previous_string = ""
    while previous_string != input_string:
        previous_string = input_string
        input_string = re.sub(r'\|\s*([-\s–—\u2010\u2212\u2012\u2015\u00AD\u2011]+)\s*\|',
                              '|', input_string)
    cleaned_string = re.sub(r'\s+', ' ', input_string)
    return cleaned_string


def select_best_ext(ext_list, text, select_model, select_tokenizer):
    ext_scores = []
    for each_ext in ext_list:
        batch_input = []
        for each_atom in each_ext["atomic_facts"]:
            batch_input.append((text, each_atom["relation"]))
        for each_disc in each_ext["discourse relations"]:
            batch_input.append((text, each_disc["relation"]))
        if len(batch_input) > 0:
            tmp_scores = sentence_cls_score(batch_input, select_model,
                                            select_tokenizer)[:, 0].cpu().numpy().tolist()
            for each_atom, score in zip(each_ext["atomic_facts"],
                                        tmp_scores[:len(each_ext["atomic_facts"])]):
                each_atom["cls conf"] = score

            for each_disc, score in zip(each_ext["discourse relations"],
                                        tmp_scores[len(each_ext["atomic_facts"]):]):
                each_disc["cls conf"] = score
            ext_scores.append([each_ext, np.sum(tmp_scores) / len(tmp_scores)])
    return ext_scores


def add_annotations_for_pronouns(text, cor_clusters):
    modifications = []
    for cluster in cor_clusters:
        main_start = cluster["main"]["start"]
        main_end = cluster["main"]["end"]
        main_text = cluster["main"]["text"]
        for mention in cluster["mentions"]:
            mention_start = mention["start"]
            mention_end = mention["end"]
            if (main_start <= mention_start and mention_end <= main_end) or (mention["text"] == main_text):
                continue
            possessive_suffix = "'" if main_text.endswith("s") else "'s"
            if mention["text"].lower() in ["his", "her", "its", "their"]:
                annotation = f"{main_text}{possessive_suffix}"
            else:
                annotation = main_text
            insert_text = f" ({annotation})"
            modifications.append((mention["end"], insert_text))
    modifications.sort(key=lambda x: x[0], reverse=True)
    for position, insert_text in modifications:
        text = text[:position] + insert_text + text[position:]
    return text


def process_text(text):
    doc = nlp(text)
    cor_clusters = []
    for cluster in doc._.coref_clusters:
        cluster_info = {
            "main": {"text": cluster.main.text, "start": cluster.main.start_char, "end": cluster.main.end_char},
            "mentions": [{"text": mention.text, "start": mention.start_char, "end": mention.end_char} for mention in cluster.mentions]
        }
        cor_clusters.append(cluster_info)
    processed_txt = add_annotations_for_pronouns(text, cor_clusters)
    return {"original_text": text, "processed_txt": processed_txt, "cor_clusters": cor_clusters}


def get_text_chunks(input_txt):
    with MosesDetokenizer('en') as detokenize:
        doc = detokenize(input_txt.split())
        sentences = sent_tokenize(doc)
        sentence_group = [[]]
        group_id = 0
        tmp_len = 0
        max_len = 256
        stride_ratio = 128
        for each_sent in sentences:
            sent_len = len(each_sent.split())
            # Slide window, until length > max_len
            if sent_len + tmp_len > max_len:
                group_len = len(sentence_group[group_id])
                if group_len < 2:
                    appending_group = []
                elif group_len < stride_ratio:
                    appending_group = [sentence_group[group_id][-1]]
                else:
                    appending_group = sentence_group[group_id][-(group_len // stride_ratio):]
                group_id += 1
                sentence_group.append(appending_group)
                tmp_len = 0
            sentence_group[group_id].append(each_sent)
            tmp_len += sent_len
        final_result = []
        for each_group in sentence_group:
            final_result.append(" ".join(each_group))
        return final_result


def evaluate_extractions_with_segments(extractions, doc_segments, tokenizer, model, device='cuda', batch_size=16):
    max_scores = [0] * len(extractions)
    for i in range(0, len(extractions)):
        tmp_ext = extractions[i]
        pairs = [(segment, tmp_ext) for segment in doc_segments]
        scores = []
        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start:batch_start + batch_size]
            texts, hypotheses = zip(*batch)
            inputs = tokenizer(texts, hypotheses, padding=True, truncation="longest_first",
                               max_length=1280, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            softmax_scores = torch.softmax(outputs.logits, dim=1)
            batch_scores = softmax_scores[:, 0].tolist()
            # Assume 0 is the index for "entailment"
            scores.extend(batch_scores)
        max_scores[i] = max(scores) if scores else 0
    return max_scores


def main():
    all_input = []
    device = 'cuda'
    tokenizer, model = load_model_and_tokenizer(model_name="Inria-CEDAR/FDSpotter-DeBERTa-V3-Large", device=device)
    
    with open("aggre_fact.csv", mode='r', newline='', encoding='utf-8') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            all_input.append(row)
    print(all_input[-1])

    ext_dict = {}
    with open("out0_gpt4o_ext_prob.json", 'r', encoding='utf-8-sig') as f:
        tmp_all_ext = json.load(f)
        for each_entry in tqdm(tmp_all_ext):
            tmp_txt = each_entry["txt"]
            tmp_extractions = [{"atomic facts": x["atomic facts"],
                                "discourse relations": x["discourse relations"]}
                               for x in each_entry["ext"]]
            processed_ext = []
            for each_ext in tmp_extractions:
                tmp_atom = each_ext["atomic facts"]
                tmp_disc = each_ext["discourse relations"]
                tmp_atom = assemble_gpt_conf(tmp_atom)
                tmp_disc = assemble_gpt_conf(tmp_disc)
                processed_ext.append({"atomic_facts": tmp_atom, "discourse relations": tmp_disc})
            if tmp_txt not in ext_dict:
                ext_dict[tmp_txt] = processed_ext
            else:
                ext_dict[tmp_txt].extend(processed_ext)
    with open('aggrefact/ext_gpt4o_aggregated_prob.json', 'w', encoding='utf-8-sig') as file:
        json.dump(ext_dict, file, ensure_ascii=False, indent=2)

    score_json = "out1_intrinsic_extrinsic_confidence.json"
    output_dict = {}

    for item in tqdm(all_input):
        summary = item["summary"]
        doc_segments = get_text_chunks(process_text(item["doc"])["processed_txt"])
        tmp_all_ext = ext_dict[summary]
        ext_scores = select_best_ext(tmp_all_ext, summary, model, tokenizer)
        top_k_ext = sorted(ext_scores, key=lambda x: x[-1], reverse=True)
        for each_ext in top_k_ext:
            ext_lines = []
            ext_conf = []
            tmp_ref = []
            for each_fact in each_ext[0]['atomic_facts']:
                ext_lines.append(each_fact["relation"])
                ext_conf.append(each_fact["cls conf"])
                tmp_ref.append(each_fact)
            for each_disc in each_ext[0]["discourse relations"]:
                ext_lines.append(each_disc["relation"])
                ext_conf.append(each_disc["cls conf"])
                tmp_ref.append(each_disc)
            max_scores = evaluate_extractions_with_segments(
                ext_lines, doc_segments, tokenizer, model, device=device)
            for entity, score in zip(tmp_ref, max_scores):
                entity["cls faith"] = score
            if len(max_scores) == 0:
                item["FactOverlapGPT4_score"] = 0
            else:
                ext_conf = torch.tensor(ext_conf, device=device)
                item["FactOverlapGPT4_score"] = float(
                    torch.dot(ext_conf,
                              # torch.log(torch.tensor(max_scores, device=device))
                              torch.tensor(max_scores, device=device)
                              ) / torch.sum(ext_conf))
            item["FactOverlapGPT4_label"] = ""
        # print(summary)
        # print(top_k_ext)
        output_dict[summary] = top_k_ext
    with open(score_json, 'w', encoding='utf-8-sig') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
