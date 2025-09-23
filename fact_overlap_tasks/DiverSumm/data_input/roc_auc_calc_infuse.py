import csv
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def compute_roc_auc_for_origins(csv_path):
    data_by_origin = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            origin = row["origin"].strip()
            label_str = row["label"].strip()
            scores_str = row["scores"].strip()
            label = int(label_str)
            score_list = scores_str.split()
            float_scores = [float(s) for s in score_list]
            mean_score = np.mean(float_scores)
            data_by_origin[origin].append((label, mean_score))

    for origin, items in data_by_origin.items():
        labels = [it[0] for it in items]
        mean_scores = [it[1] for it in items]

        if len(set(labels)) < 2:
            print(f"Origin '{origin}' has only one label, can't calculate ROC-AUC")
            continue

        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        auc_weighted = roc_auc_score(labels, mean_scores, )

        print(f"Origin = {origin}, AUC = {auc_weighted:.4f}")


if __name__ == "__main__":
    csv_file_path = "DiverSumm1.5.csv"
    compute_roc_auc_for_origins(csv_file_path)
