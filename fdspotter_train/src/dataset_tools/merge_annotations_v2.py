import os
import json
from collections import defaultdict

keywords = [
    'neg_condition', 'concession', 'condition', 'contrast', 'disjunctive',
    'location', 'precedence', 'reason', 'result', 'succession', "synchronous"
]
folders = [
    '../../data/build/disc_nli/test_annotations/ann_lingnli',
    '../../data/build/disc_nli/test_annotations/ann_mnli',
    '../../data/build/disc_nli/test_annotations/ann_wanli', ]

group_cmp = ['concession', 'contrast', ]
group_contingency = ['reason', 'result', 'neg_condition', 'condition', ]
group_tmp = ['precedence', 'succession', "synchronous"]

list_cmp = []
list_contingency = []
list_tmp = []

content_dict = {key: [] for key in keywords}
uid_counts = defaultdict(int)

for folder in folders:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            for line in file:
                json_obj = json.loads(line)
                original_uid = json_obj.get('uid', None)
                if "lingnli" in folder:
                    original_uid = "lingnli" + original_uid.split("jsonl")[-1]
                elif "mnli" in folder:
                    original_uid = "mmnli" + original_uid
                if original_uid:
                    uid_counts[original_uid] += 1
                    new_uid = f"{original_uid}{chr(96 + uid_counts[original_uid])}"
                    json_obj['uid'] = new_uid
                if "lingnli" in folder:
                    json_obj["source"] = "lingnli"
                elif "wanli" in folder:
                    json_obj["source"] = "wanli"
                elif "mnli" in folder:
                    json_obj["source"] = "mnli"
                for each_conn in json_obj["original_conn"]:
                    if each_conn == "alternative":
                        pass
                    elif each_conn == "disjunctive":
                        each_conn = "neg_condition"
                        json_obj["original_conn"] = ["neg_condition"]
                        print(json_obj["original_conn"])

                    if each_conn in keywords:
                        content_dict[each_conn].append(json_obj)

                    if each_conn in group_tmp:
                        if len(list_tmp) != 0:
                            if list_tmp[-1]["hypothesis"] != json_obj["hypothesis"]:
                                list_tmp.append(json_obj)
                        else:
                            list_tmp.append(json_obj)
                    elif each_conn in group_contingency:
                        if len(list_contingency) != 0:
                            if list_contingency[-1]["hypothesis"] != json_obj["hypothesis"]:
                                list_contingency.append(json_obj)
                        else:
                            list_contingency.append(json_obj)
                    elif each_conn in group_cmp:
                        if len(list_tmp) == 0:
                            if list_cmp[-1]["hypothesis"] != json_obj["hypothesis"]:
                                list_cmp.append(json_obj)
                        else:
                            list_cmp.append(json_obj)

for keyword, json_objects in content_dict.items():
    of_all = f"../../data/build/disc_nli/test_annotations/merged/{keyword}_merged.jsonl"
    of_gen = f"../../data/build/disc_nli/test_annotations/generated/{keyword}_gen.jsonl"
    of_orig = f"../../data/build/disc_nli/test_annotations/original/{keyword}_original.jsonl"
    sorted_objects = sorted(json_objects, key=lambda x: x['uid'])
    with open(of_all, 'w', encoding='utf-8-sig') as w_all, \
            open(of_gen, 'w', encoding='utf-8-sig') as w_gen, \
            open(of_orig, 'w', encoding='utf-8-sig') as w_orig:
        for json_obj in json_objects:
            json_str = json.dumps(json_obj, ensure_ascii=False)
            w_all.write(json_str + '\n')
            if 'modified_conn' in json_obj:
                w_gen.write(json_str + '\n')
            else:
                w_orig.write(json_str + '\n')

    files = {
        'tmp_with_modified': '../../data/build/disc_nli/test_annotations/generated/temporal_gen.jsonl',
        'tmp_without_modified': '../../data/build/disc_nli/test_annotations/merged/temporal_orig.jsonl',
        'contingency_with_modified': '../../data/build/disc_nli/test_annotations/generated/contingency_gen.jsonl',
        'contingency_without_modified': '../../data/build/disc_nli/test_annotations/merged/contingency_orig.jsonl',
        'cmp_with_modified': '../../data/build/disc_nli/test_annotations/generated/comparison_gen.jsonl',
        'cmp_without_modified': '../../data/build/disc_nli/test_annotations/merged/comparison_orig.jsonl'
    }

    with open(files['tmp_with_modified'], 'w', encoding='utf-8-sig') as tmp_with, \
            open(files['tmp_without_modified'], 'w', encoding='utf-8-sig') as tmp_without, \
            open(files['contingency_with_modified'], 'w', encoding='utf-8-sig') as contingency_with, \
            open(files['contingency_without_modified'], 'w', encoding='utf-8-sig') as contingency_without, \
            open(files['cmp_with_modified'], 'w', encoding='utf-8-sig') as cmp_with, \
            open(files['cmp_without_modified'], 'w', encoding='utf-8-sig') as cmp_without:

        for json_obj in list_tmp:
            json_str = json.dumps(json_obj, ensure_ascii=False)
            if 'modified_conn' in json_obj:
                tmp_with.write(json_str + '\n')
            else:
                tmp_without.write(json_str + '\n')

        for json_obj in list_contingency:
            json_str = json.dumps(json_obj, ensure_ascii=False)
            if 'modified_conn' in json_obj:
                contingency_with.write(json_str + '\n')
            else:
                contingency_without.write(json_str + '\n')

        for json_obj in list_cmp:
            json_str = json.dumps(json_obj, ensure_ascii=False)
            if 'modified_conn' in json_obj:
                cmp_with.write(json_str + '\n')
            else:
                cmp_without.write(json_str + '\n')

    print("Processing complete. Files have been saved.")
