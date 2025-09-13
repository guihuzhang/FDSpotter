import csv
import json
import re
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# nltk.download('stopwords')
# nltk.download('punkt')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')


def word_coverage(txt, ext):
    # ext covers how much percent of words in txt
    ext = " ".join(ext)
    txt = re.sub(r'[^\w\s]', ' ', txt.replace('|', ' '))
    txt_words = word_tokenize(txt)
    txt_word_set = set([word.lower() for word in txt_words if word.lower() not in stop_words])
    ext = re.sub(r'[^\w\s]', ' ', ext.replace('|', ' '))
    ext_words = word_tokenize(ext)
    ext_word_set = set([word.lower() for word in ext_words if word.lower() not in stop_words])
    intersection = ext_word_set.intersection(txt_word_set)
    len_txt = len(txt_word_set)
    if len_txt == 0:
        return 0.
    else:
        return len(intersection) / len_txt


def remove_subsumed_ext(ext, conf, faith):
    to_remove = set()
    for i in range(len(ext)):
        for j in range(len(ext)):
            if i != j and ext[i] in ext[j]:
                to_remove.add(i)
    filtered_ext = [ext[i] for i in range(len(ext)) if i not in to_remove]
    filtered_conf = [conf[i] for i in range(len(conf)) if i not in to_remove]
    filtered_faith = [faith[i] for i in range(len(faith)) if i not in to_remove]
    return filtered_ext, filtered_conf, filtered_faith


def main():
    all_input = []
    with open("./aggre_fact.csv", mode='r', newline='', encoding='utf-8') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            all_input.append(row)
    print(all_input[-1])
    # the chunks of texts
    with open("./out1_intrinsic_extrinsic_confidence.json", 'r', encoding='utf-8-sig') as f:
        ext_conf_faith = json.load(f)
    use_ext_num = 1
    consider_ext = 5
    output_csv_path = './out2_aggre_fact_overlap_gpt4o_plogp_binary.5.csv'
    min_score = 0
    with_disc_count = 0
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as out_file:
        fieldnames = reader.fieldnames + ['FactOverlapGPT4_score', 'FactOverlapGPT4_label']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in tqdm(all_input):
            item["FactOverlapGPT4_score"] = 0
            summary = item["summary"]
            tmp_extractions = ext_conf_faith[summary][:consider_ext]
            # print(tmp_extractions)
            ext_scores = []
            for each_ext in tmp_extractions:
                tmp_conf = []
                for each_atom in each_ext[0]["atomic_facts"]:
                    tmp_conf.append(each_atom["cls conf"])
                for each_disc in each_ext[0]["discourse relations"]:
                    tmp_conf.append(each_disc["cls conf"])
                ext_scores.append([each_ext[0], np.sum(tmp_conf) / len(tmp_conf)])
            # print(ext_scores)
            top_k_ext = sorted(ext_scores, key=lambda x: x[-1], reverse=True)[:use_ext_num]
            sum_scores = 0
            sum_weights = 0
            line_num = 0
            has_disc = False
            for tmp_ext, ext_score in top_k_ext:
                for each_atom in tmp_ext["atomic_facts"]:
                    if each_atom["cls conf"] >= 0.5:
                        sum_scores += np.log(each_atom["cls faith"])
                        sum_weights += 1
                        line_num += 1
                for each_disc in tmp_ext["discourse relations"]:
                    if each_disc["cls conf"] >= 0.5:
                        sum_scores += np.log(each_disc["cls faith"])
                        sum_weights += 1
                        line_num += 1
                        has_disc = True
            if has_disc:
                with_disc_count += 1
            # print(sum_scores, sum_weights)
            if line_num > 0:
                item["FactOverlapGPT4_score"] = sum_scores / sum_weights
                if item["FactOverlapGPT4_score"] < min_score:
                    min_score = item["FactOverlapGPT4_score"]
            else:
                print(summary)
                print(item["label"])
            item["FactOverlapGPT4_label"] = ""
            writer.writerow(item)
            out_file.flush()


if __name__ == '__main__':
    main()
