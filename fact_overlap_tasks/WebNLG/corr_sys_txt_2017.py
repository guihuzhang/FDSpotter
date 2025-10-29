from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import json
import random
import sys
import datetime
import math
from tqdm import tqdm
random.seed(10)


def correlations_bootstrapping(folder1, folder2, selected_bootstraping, logging_path, bootstrapping):
    wb = open(logging_path, "a")
    subfiles = [f.path for f in os.scandir(folder2)]
    scores_classifier = []
    scores_human = []
    count = 0
    for subfile in tqdm(subfiles, desc="system level systems"):
        path_human = folder1 + subfile.split('/')[-1]
        rb = open(path_human)
        scores_h = [float(line.strip()) for line in rb.readlines()]
        scores_human.append(scores_h)
        rb.close()

        rb = open(subfile, "r")
        scores_c = [float(line.strip()) for line in rb.readlines()]
        scores_classifier.append(scores_c)
        rb.close()
        count += 1

    pear_boot = []
    sp_boot = []
    k_boot = []
    iter = 0
    while iter < bootstrapping:
        selected = selected_bootstraping[iter]
        iter += 1
        scores_h = []
        scores_c = []
        for j in range(len(scores_classifier)):
            selected_classifier = [scores_classifier[j][k] for k in selected]
            selected_human = [scores_human[j][k] for k in selected]
            scores_c.append(sum(selected_classifier) / len(selected_classifier))
            scores_h.append(sum(selected_human) / len(selected_human))

        pear = pearsonr(scores_c, scores_h)
        if pear[1] < 0.05:
            pear_boot.append(pear[0])
        sp = spearmanr(scores_c, scores_h)
        if sp[1] < 0.05:
            sp_boot.append(sp[0])
        k = kendalltau(scores_c, scores_h)
        if k[1] < 0.05:
            k_boot.append(k[0])

    if len(pear_boot):
        sorted_coef = sorted(pear_boot)
        wb.write("mean pearson: " + str(sum(pear_boot) / len(pear_boot)) + ", samples " + str(
            len(pear_boot)) + " " + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef) * 0.05)]) + ", " + str(
            sorted_coef[math.floor(len(sorted_coef) * 0.95)]) + "\n")

    if len(sp_boot):
        sorted_coef = sorted(sp_boot)
        wb.write("mean spearman: " + str(sum(sp_boot) / len(sp_boot)) + ", samples " + str(
            len(sp_boot)) + " " + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef) * 0.05)]) + ", " + str(
            sorted_coef[math.floor(len(sorted_coef) * 0.95)]) + "\n")

    if len(k_boot):
        sorted_coef = sorted(k_boot)
        wb.write("mean kendal: " + str(sum(k_boot) / len(k_boot)) + ", samples " + str(
            len(sp_boot)) + " " + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef) * 0.05)]) + ", " + str(
            sorted_coef[math.floor(len(sorted_coef) * 0.95)]) + "\n")

    wb.flush()
    wb.close()


def correlations_sentence_bootstrapping(folder1, folder2, selected_bootstraping, logging_path):
    wb = open(logging_path, "a")
    subfiles = [f.path for f in os.scandir(folder2)]
    scores_classifier = []
    scores_human = []
    count = 0
    for subfile in tqdm(subfiles, desc="sentence level systems"):
        path_human = folder1 + subfile.split('/')[-1]
        rb = open(path_human)
        scores_h = [float(line.strip()) for line in rb.readlines()]
        scores_human.append(scores_h)
        rb.close()

        rb = open(subfile, "r")
        scores_c = [float(line.strip()) for line in rb.readlines()]
        scores_classifier.append(scores_c)

        rb.close()
        count+=1

    pear_boot = []
    sp_boot = []
    k_boot = []
    iter = 0
    while iter < 1000:
        selected = selected_bootstraping[iter]
        iter +=1
        pear_boot_iter = []
        sp_boot_iter = []
        k_boot_iter = []
        for j in range(len(scores_classifier)):
            selected_classifier = [scores_classifier[j][k] for k in selected]
            selected_human = [scores_human[j][k] for k in selected]

            pear = pearsonr(selected_classifier, selected_human)
            if pear.pvalue < 0.05:
                pear_boot_iter.append(pear.statistic)
            sp = spearmanr(selected_classifier, selected_human)
            if sp.pvalue < 0.05:
                sp_boot_iter.append(sp.correlation)
            k = kendalltau(selected_classifier, selected_human)
            if k.pvalue < 0.05:
                k_boot_iter.append(k.correlation)

        if len(pear_boot_iter):
            pear_boot.append(sum(pear_boot_iter)/len(pear_boot_iter))
        if len(sp_boot_iter):
            sp_boot.append(sum(sp_boot_iter)/len(sp_boot_iter))
        if len(k_boot_iter):
            k_boot.append(sum(k_boot_iter)/len(k_boot_iter))

    if len(pear_boot):
        sorted_coef = sorted(pear_boot)
        wb.write("mean pearson: "+ str(sum(pear_boot)/len(pear_boot)) + ", samples " +str(len(pear_boot))+" "  + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef)*0.05)]) + ", "+ str(sorted_coef[math.floor(len(sorted_coef)*0.95)]) + "\n")

    if len(sp_boot):
        sorted_coef = sorted(sp_boot)
        wb.write("mean spearman: " + str(sum(sp_boot) / len(sp_boot))+ ", samples " +str(len(sp_boot))+" "  +  ", interval: " + str(sorted_coef[math.floor(len(sorted_coef)*0.05)]) + ", "+ str(sorted_coef[math.floor(len(sorted_coef)*0.95)]) + "\n")

    if len(k_boot):
        sorted_coef = sorted(k_boot)
        wb.write("mean kendal: " + str(sum(k_boot) / len(k_boot))+ ", samples " +str(len(sp_boot))+" "  + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef)*0.05)]) + ", "+ str(sorted_coef[math.floor(len(sorted_coef)*0.95)]) + "\n")


def correlations_text(folder1, folder2, selected_bootstraping, logging_path):
    wb = open(logging_path, "a")
    subfiles = [f.path for f in os.scandir(folder2)]
    scores_classifier = []
    scores_human = []
    count = 0
    size_test = 0
    for subfile in tqdm(subfiles, desc="text level systems"): #a file per system
        path_human = folder1 + subfile.split('/')[-1]
        rb = open(path_human)
        scores_h = [float(line.strip()) for line in rb.readlines()]
        scores_human.append(scores_h)
        rb.close()

        rb = open(subfile, "r")
        scores_c = [float(line.strip()) for line in rb.readlines()]
        scores_classifier.append(scores_c)
        size_test=len(scores_c)
        rb.close()
        count+=1

    pear_boot = []
    sp_boot = []
    k_boot = []
    iter = 0
    lens = []
    while iter < 1000:
        selected = selected_bootstraping[iter]
        iter += 1
        pear_iter = []
        sp_iter = []
        k_iter = []
        for j in range(size_test):
            pos = selected[j]
            scores_c = [scores_classifier[k][pos] for k in range(count)]
            scores_h = [scores_human[k][pos] for k in range(count)]

            pear = pearsonr(scores_c, scores_h)
            if pear.pvalue < 0.05:
                pear_iter.append(pear.statistic)
            sp = spearmanr(scores_c, scores_h)
            if sp.pvalue < 0.05:
                sp_iter.append(sp.correlation)
            k = kendalltau(scores_c, scores_h)
            if k.pvalue < 0.05:
                k_iter.append(k.correlation)

        if len(pear_iter):
            pear_boot.append(sum(pear_iter)/len(pear_iter))
            lens.append(len(pear_iter))
        if len(sp_iter):
            sp_boot.append(sum(sp_iter)/len(sp_iter))
        if len(k_iter):
            k_boot.append(sum(k_iter)/len(k_iter))

    if len(pear_boot):
        sorted_coef = sorted(pear_boot)
        wb.write("mean pearson: "+ str(sum(pear_boot)/len(pear_boot)) + ", samples " +str(len(pear_boot))+" "  + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef)*0.05)]) + ", "+ str(sorted_coef[math.floor(len(sorted_coef)*0.95)]) + "\n")

    if len(sp_boot):
        sorted_coef = sorted(sp_boot)
        wb.write("mean spearman: " + str(sum(sp_boot) / len(sp_boot))+ ", samples " +str(len(sp_boot))+" "  +  ", interval: " + str(sorted_coef[math.floor(len(sorted_coef)*0.05)]) + ", "+ str(sorted_coef[math.floor(len(sorted_coef)*0.95)]) + "\n")

    if len(k_boot):
        sorted_coef = sorted(k_boot)
        wb.write("mean kendal: " + str(sum(k_boot) / len(k_boot))+ ", samples " +str(len(sp_boot))+" "  + ", interval: " + str(sorted_coef[math.floor(len(sorted_coef)*0.05)]) + ", "+ str(sorted_coef[math.floor(len(sorted_coef)*0.95)]) + "\n")


if __name__ == '__main__':
    parameters = json.load(open(sys.argv[1]))

    selected_bootstrapping = []
    indexes = [i for i in range(parameters['no-samples'])]

    for iter in range(parameters["bootstrapping"]):
        selected = random.choices(indexes, k=parameters['no-samples'])
        selected_bootstrapping.append(selected)

    wb = open(parameters["logging-path"], "a")
    now = datetime.datetime.now()
    wb.write("\n\n")
    wb.write(str(now) + "\n")
    wb.flush()

    # wb.write("System level human - human\n")
    # wb.flush()
    # print("sys human aggrements!")
    # for value1 in tqdm(parameters['human-categories'], desc="dimensions! "):
    #     for value2 in parameters['human-categories']:
    #         wb.write(value1.upper() + " - " + value2.upper() + "\n")
    #         wb.flush()
    #         folder2 = parameters['human-annotations'] + "/" + value1 + "/"
    #
    #         folder1 = parameters['human-annotations'] + "/" + value2 + "/"
    #
    #         correlations_bootstrapping(folder1, folder2, selected_bootstrapping, parameters["logging-path"], parameters["bootstrapping"])

    wb.write("\n System level human - metrics\n")
    wb.flush()
    print("system level!")
    for value in tqdm(parameters['human-categories'], desc="dimensions! "):
        wb.write(value.upper() + '\n')
        folder1 = parameters['human-annotations'] + "/" + value + "/"
        # subfolders = [f.path for f in os.scandir(parameters["metrics-path"]) if f.is_dir()]
        subfolders = ["metrics_results/webnlg2017/fact-overlap-large523", "metrics_results/webnlg2017/fact-spotter", ]
        for i in tqdm(range(len(subfolders)), desc="systems! "):
            wb.write(value.upper() + " " + subfolders[i] + "\n")
            wb.flush()
            correlations_bootstrapping(folder1, subfolders[i], selected_bootstrapping, parameters["logging-path"],
                                       parameters["bootstrapping"])

    wb.write("Sentence 1 level human - human\n")
    wb.flush()
    print("txt level v1 human!")
    for value1 in tqdm(parameters['human-categories'], desc="dimensions! "):
        for value2 in parameters['human-categories']:
            wb.write(value1.upper() + " - " + value2.upper() + "\n")
            wb.flush()
            folder2 = parameters['human-annotations'] + "/" + value1 + "/"

            folder1 = parameters['human-annotations'] + "/" + value2 + "/"

            correlations_sentence_bootstrapping(
                folder1, folder2, selected_bootstrapping, parameters["logging-path"])

    wb.write("\n Sentence 1 level human - metrics\n")
    wb.flush()
    print("txt level v1!")
    for value in tqdm(parameters['human-categories'], desc="dimensions! "):
        wb.write(value.upper() + '\n')
        folder1 = parameters['human-annotations'] + "/" + value + "/"
        # subfolders = [f.path for f in os.scandir(parameters["metrics-path"]) if f.is_dir()]
        subfolders = ["metrics_results/webnlg2017/fact-overlap-large523", "metrics_results/webnlg2017/fact-spotter", ]
        for i in tqdm(range(len(subfolders)), desc="systems! "):
            wb.write(value.upper() + " " + subfolders[i] + "\n")
            wb.flush()
            correlations_sentence_bootstrapping(folder1, subfolders[i], selected_bootstrapping, parameters["logging-path"])

    # wb.write("Sentence 2 level human - human\n")
    # wb.flush()
    # print("txt level v2 human!")
    # for value1 in tqdm(parameters['human-categories'], desc="dimensions! "):
    #     for value2 in parameters['human-categories']:
    #         wb.write(value1.upper() + " - " + value2.upper() + "\n")
    #         wb.flush()
    #         folder2 = parameters['human-annotations'] + "/" + value1 + "/"
    #
    #         folder1 = parameters['human-annotations'] + "/" + value2 + "/"
    #
    #         correlations_text(folder1, folder2, selected_bootstrapping, parameters["logging-path"])

    wb.write("\n Sentence 2 level human - metrics\n")
    wb.flush()
    print("txt level v2!")
    for value in tqdm(parameters['human-categories'], desc="dimensions! "):
        wb.write(value.upper() + '\n')
        folder1 = parameters['human-annotations'] + "/" + value + "/"
        # subfolders = [f.path for f in os.scandir(parameters["metrics-path"]) if f.is_dir()]
        subfolders = ["metrics_results/webnlg2017/fact-overlap-large523", "metrics_results/webnlg2017/fact-spotter", ]
        for i in tqdm(range(len(subfolders)), desc="systems!"):
            wb.write(value.upper() + " " + subfolders[i] + "\n")
            wb.flush()
            correlations_text(folder1, subfolders[i], selected_bootstrapping, parameters["logging-path"])
