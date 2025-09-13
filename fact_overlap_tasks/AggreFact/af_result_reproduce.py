import pandas as pd
import numpy as np
import sklearn
from utils import choose_best_threshold
import warnings
warnings.filterwarnings('ignore')

from utils import resample_balanced_acc
from utils import SOTA, XFORMER, OLD, MAPPING
import re


def extract_number(tensor_string):
    match = re.search(r'tensor\(([^,]+), device=', tensor_string)
    return float(match.group(1)) if match else None


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("out2_aggre_fact_overlap_gpt4o_plogp_binary.5.csv")
# df['FactOverlapGPT4_score'] = df['FactOverlapGPT4_score'].apply(np.exp)

# split data
df_val = df[df.cut == 'val']
df_val_sota = df_val[df_val.model_name.isin(SOTA)]

df_test = df[df.cut == 'test']
df_test_sota = df_test[df_test.model_name.isin(SOTA)]

dataset_list = ['XSumFaith', 'Polytope', 'FactCC', 'SummEval', 'FRANK', 'Wang20', 'CLIFF', 'Goyal21', 'Cao22']
systems = ['DAE', 'QuestEval', 'SummaC-ZS', 'SummaC-Conv', 'QAFactEval', "FactOverlapGPT4", ]
origins = ['cnndm', 'xsum']

main_df = pd.DataFrame(
    columns=['system', 'origin', 'count', 'dataset', 'category', 'bl_acc']
)

results = []

for system in systems:
    df[f'{system}_label'] = None

for system in systems:
    for origin in origins:
        for dataset in dataset_list:
            for i, model_novelty in enumerate([SOTA, XFORMER, OLD]):

                df_val_temp = df_val[
                    (df_val.dataset == dataset) & (df_val.origin == origin) & (df_val.model_name.isin(model_novelty))]

                df_test_temp = df_test[(df_test.dataset == dataset) & (df_test.origin == origin) & (
                    df_test.model_name.isin(model_novelty))]

                if len(df_val_temp) > 0 and len(df_test_temp) > 0:
                    best_thresh, best_f1 = choose_best_threshold(
                        df_val_temp.label.values, df_val_temp[f'{system}_score'].values)

                    scores_test = df_test_temp[f'{system}_score'].values

                    preds_test = [1 if score > best_thresh else 0 for score in scores_test]
                    df.loc[df_test_temp.index, f'{system}_label'] = preds_test

                    balanced_acc = sklearn.metrics.balanced_accuracy_score(df_test_temp.label.values, preds_test)

                    main_df.loc[len(main_df.index)] = [
                        system, origin, len(preds_test), dataset, MAPPING[i], balanced_acc
                    ]

                    results.append({"system": system, "dataset_name": dataset, 'origin': origin,
                                    'count': len(scores_test), 'cat': MAPPING[i], "labels": df_test_temp.label.values,
                                    "preds": preds_test, "scores": scores_test})

df = df.reindex(
    columns=['dataset', 'origin', 'id', 'doc', 'summary', 'model_name', 'label', 'cut', 'DAE_score', 'DAE_label',
             'QuestEval_score', 'QuestEval_label', 'SummaC-ZS_score', 'SummaC-ZS_label',
             'SummaC-Conv_score', 'SummaC-Conv_label', 'QAFactEval_score', 'QAFactEval_label',
             'FactOverlapGPT4_score', 'FactOverlapGPT4_label', ])

# Table 8
main_df_pivot_bacc = main_df.pivot(index=['origin', 'dataset', 'category', 'count'], columns='system', values='bl_acc')
main_df_pivot_bacc = main_df_pivot_bacc.reindex(columns=systems)
main_df_pivot_bacc.round(3)
print(main_df_pivot_bacc)

# Table 4
scores = []
for cat in MAPPING.values():
    score = []
    for system in systems:
        system_df = main_df[(main_df.system == system) & (main_df.category == cat) & (main_df.origin == 'cnndm')]
        value = sum(system_df['count'] * system_df['bl_acc']) / sum(system_df['count'])
        score.append(value)
    scores.append(score)

weighted_df = pd.DataFrame(
    scores,
    columns=systems,
    index=['SOTA', 'XFORMER', 'OLD']
)

average_scores = weighted_df.mean(axis=0)
average_row = pd.DataFrame([average_scores], columns=systems, index=['Average'])
final_df = pd.concat([weighted_df, average_row])
print(final_df)

# Table 4
scores = []
for cat in MAPPING.values():
    score = []
    for system in systems:
        system_df = main_df[(main_df.system == system) & (main_df.category == cat) & (main_df.origin == 'xsum')]
        value = sum(system_df['count'] * system_df['bl_acc']) / sum(system_df['count'])
        score.append(value)
    scores.append(score)
weighted_df = pd.DataFrame(scores, columns=systems, index=['SOTA', 'XFORMER', 'OLD'])
average_scores = weighted_df.mean(axis=0)
average_row = pd.DataFrame([average_scores], columns=systems, index=['Average'])
final_df = pd.concat([weighted_df, average_row])
print(final_df)

grouped_results = {}
for sys in systems:
    grouped_results[sys] = {}
    for origin in origins:
        grouped_results[sys][origin] = {}
        for cat in ['SOTA', 'XFORMER', 'OLD']:
            grouped_results[sys][origin][cat] = {'preds': [], 'labels': []}

for res in results:
    system = res['system']
    category = res['cat']
    origin = res['origin']
    grouped_results[system][origin][category]['preds'].extend(res['preds'])
    grouped_results[system][origin][category]['labels'].extend(res['labels'])

P5 = 5 / 2
P1 = 1 / 2

# for system, origins in grouped_results.items():
#     for origin, cats in origins.items():
#         for category, data in cats.items():
#             preds = data['preds']
#             labels = data['labels']
#             if preds and labels:
#                 bacc = sklearn.metrics.balanced_accuracy_score(labels, preds)
#                 samples = resample_balanced_acc(data['preds'], data['labels'])
#                 low5, high5 = np.percentile(samples, P5), np.percentile(samples, 100 - P5)
#                 low1, high1 = np.percentile(samples, P1), np.percentile(samples, 100 - P1)
#                 print(f"{system} | {category} | {origin}: BAcc - Low5 = {bacc - low5:.3f}")

main_sota_df = pd.DataFrame(
    columns=['system', 'origin', 'bl_acc']
)

results = []

for system in systems:
    for origin in origins:
        df_val_temp = df_val_sota[(df_val_sota.origin == origin)]
        df_test_temp = df_test_sota[(df_test_sota.origin == origin)]

        best_thresh, best_f1 = choose_best_threshold(df_val_temp.label.values, df_val_temp[f'{system}_score'].values)

        scores_test = df_test_temp[f'{system}_score'].values
        preds_test = [1 if score > best_thresh else 0 for score in scores_test]

        f1_score = sklearn.metrics.balanced_accuracy_score(df_test_temp.label.values, preds_test)

        main_sota_df.loc[len(main_sota_df.index)] = [
            system, origin, f1_score
        ]

        results.append({"system": system, 'origin': origin, "labels": df_test_temp.label.values,
                        "preds": preds_test, "scores": scores_test})

# Table 5
# standard deviation may differ due to randomness
# from https://github.com/tingofurro/summac/
P5 = 5 / 2  # Correction due to the fact that we are running 2 tests with the same data
P1 = 1 / 2  # Correction due to the fact that we are running 2 tests with the same data
for origin in origins:
    sampled_batch_preds = {res["system"]: [] for res in results}
    for res in results:
        if res['origin'] == origin:
            samples = resample_balanced_acc(res["preds"], res["labels"])
            sampled_batch_preds[res["system"]].append(samples)
            low5, high5 = np.percentile(samples, P5), np.percentile(samples, 100 - P5)
            low1, high1 = np.percentile(samples, P1), np.percentile(samples, 100 - P1)
            bacc = sklearn.metrics.balanced_accuracy_score(res["labels"], res["preds"])
            print(res['origin'].center(6), res["system"].center(20), " %.3f, %.3f" % (bacc, bacc - low5))
    print()
