import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from tabulate import tabulate


def system_level_correlation_summeval(human_metric):
    print(f'System Level: {human_metric}')
    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']
    with open('summeval_final2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    auto_metrics = ['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo',
                    'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo',
                    'chatgpt_%s' % human_metric, "fop1logp2", ]
    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []
    for metric in auto_metrics:
        score_per_system = {}
        gt_per_system = {}
        for doc_id in data:
            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                if sys_name not in score_per_system:
                    score_per_system[sys_name] = []
                    gt_per_system[sys_name] = []
                score_per_system[sys_name].append(sys_summs[sys_name]['scores'][metric])
                gt_per_system[sys_name].append(sys_summs[sys_name]['scores'][human_metric])
        score_list = []
        gt_list = []
        for sys_name in score_per_system:
            score_list.append(sum(score_per_system[sys_name]) / len(score_per_system[sys_name]))
            gt_list.append(sum(gt_per_system[sys_name]) / len(gt_per_system[sys_name]))
        spearman = spearmanr(score_list, gt_list)[0]
        pearson = pearsonr(score_list, gt_list)[0]
        kendall = kendalltau(score_list, gt_list)[0]
        metric_with_corr.append([metric, spearman, pearson, kendall])
    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))


def dataset_level_correlation_summeval(human_metric):
    print(f'Dataset Level: {human_metric}')
    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']
    with open('summeval_final2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    auto_metrics = ['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo',
                    'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo',
                    'chatgpt_%s' % human_metric, "fop1logp2", ]
    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []
    for metric in auto_metrics:
        correlations = []
        target_scores = []
        prediction_scores = []
        for doc_id in data:
            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['scores'][metric])
                target_scores.append(sys_summs[sys_name]['scores'][human_metric])
        correlations.append([
            spearmanr(target_scores, prediction_scores)[0],
            pearsonr(target_scores, prediction_scores)[0],
            kendalltau(target_scores, prediction_scores)[0],
        ])
        corr_mat = np.array(correlations)
        spearman, pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])
    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))


def sample_level_correlation_summeval(human_metric):
    print(f'Sample Level: {human_metric}')
    assert human_metric in ['coherence', 'relevance', 'consistency', 'fluency']
    with open('summeval_final2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    test_metrics = ['rouge1_f', 'rouge2_f', 'rougel_f', 'bert_score_f', 'mover_score', 'prism_src_hypo',
                    'bart_score_src_hypo', 'bart_score_cnn_src_hypo', 'bart_score_para_src_hypo',
                    'chatgpt_%s' % human_metric, "fop1logp2", ]
    headers = ['metric', 'spearman', 'pearsonr', 'kendalltau']
    metric_with_corr = []
    for metric in test_metrics:
        correlations = []
        for doc_id in data:
            target_scores = []
            prediction_scores = []
            sys_summs = data[doc_id]['sys_summs']
            for sys_name in sys_summs:
                prediction_scores.append(sys_summs[sys_name]['scores'][metric])
                target_scores.append(sys_summs[sys_name]['scores'][human_metric])
            if len(set(prediction_scores)) == 1 or len(set(target_scores)) == 1:
                continue
            correlations.append([
                spearmanr(target_scores, prediction_scores)[0],
                pearsonr(target_scores, prediction_scores)[0],
                kendalltau(target_scores, prediction_scores)[0],
            ])
        corr_mat = np.array(correlations)
        spearman, pearman, ktau = np.mean(corr_mat[:, 0]), np.mean(corr_mat[:, 1]), np.mean(corr_mat[:, 2])
        metric_with_corr.append([metric, spearman, pearman, ktau])
    print(tabulate(metric_with_corr, headers=headers, tablefmt='simple'))


def main():
    aspect = 'consistency'
    system_level_correlation_summeval(aspect)
    dataset_level_correlation_summeval(aspect)
    sample_level_correlation_summeval(aspect)


if __name__ == '__main__':
    main()
