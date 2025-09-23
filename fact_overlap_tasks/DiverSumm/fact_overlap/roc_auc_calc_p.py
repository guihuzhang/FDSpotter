import csv
import json
import math
from collections import defaultdict

from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def load_three_jsons(json_paths):
    """Returns a dict: { summary_str: combined_list_of_extractions }. """
    combined_map = {}
    for path in json_paths:
        with open(path, 'r', encoding='utf-8-sig') as f:
            partial_map = json.load(f)
        for summ_text, exts in partial_map.items():
            summ_text = summ_text.strip()
            if summ_text not in combined_map:
                combined_map[summ_text] = []
            combined_map[summ_text].extend(exts)
    return combined_map


def compute_score(ext_conf_faith, summary, consider_ext=5, use_ext_num=1):
    extracted_list = ext_conf_faith[summary][:consider_ext]
    if not extracted_list:
        return 0.0
    # Sort by second element (some aggregator) descending
    # each item looks like: [ { "atomic_facts":..., "discourse relations":... }, average_conf_float ]
    sorted_ext = sorted(extracted_list, key=lambda x: x[-1], reverse=True)
    top_k_ext = sorted_ext[:use_ext_num]
    sum_scores = 0.0
    sum_weights = 0
    for tmp_ext, avg_conf in top_k_ext:
        # tmp_ext is e.g. { "atomic_facts": [...], "discourse relations": [...] }
        atomic_facts = tmp_ext["atomic_facts"]
        disc_rels = tmp_ext["discourse relations"]
        # only consider atomic facts that have cls conf >= 0.5
        for each_atom in atomic_facts:
            if each_atom["cls conf"] >= 0.5:
                faith_val = each_atom["cls faith"]
                sum_scores += faith_val
                sum_weights += 1
        # likewise for discourse relations
        for each_disc in disc_rels:
            if each_disc["cls conf"] >= 0.5:
                sum_scores += each_disc["cls faith"]
                sum_weights += 1
    if sum_weights > 0:
        final_score = sum_scores / sum_weights
    else:
        final_score = 0.0
    return final_score


def main():
    csv_path = "../data_input/DiverSumm.csv"
    all_rows = []
    with open(csv_path, mode='r', newline='', encoding='utf-8') as in_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames
        for row in reader:
            all_rows.append(row)

    json_paths = [
        "DSo1eval256part1.json",
        "DSo1eval256part2.json",
        "DSo1eval256part3.json"
    ]

    ext_conf_faith = load_three_jsons(json_paths)
    print("Merged JSON extractions:", len(ext_conf_faith))

    if "FactOverlapGPT4_score" not in fieldnames:
        fieldnames.append("FactOverlapGPT4_score")

    for row in tqdm(all_rows, desc="Computing GPT4-overlap scores"):
        summary = row["summary"].strip()
        # compute score
        final_score = compute_score(ext_conf_faith, summary, consider_ext=5, use_ext_num=1)
        row["FactOverlapGPT4_score"] = final_score

    #    We'll re-read or reuse 'all_rows'.
    data_by_origin = defaultdict(list)
    for row in all_rows:
        origin = row["origin"].strip()
        label_str = row["label"].strip()
        label = int(label_str) if label_str.isdigit() else 0
        score = float(row["FactOverlapGPT4_score"])
        data_by_origin[origin].append((label, score))

    for origin, items in data_by_origin.items():
        labels = [it[0] for it in items]
        scores = [it[1] for it in items]
        if len(set(labels)) < 2:
            print(f"Origin '{origin}' has only one label => can't compute ROC-AUC.")
            continue
        auc_val = roc_auc_score(labels, scores)
        print(f"Origin = {origin}, AUC = {auc_val:.4f}")


if __name__ == "__main__":
    main()
