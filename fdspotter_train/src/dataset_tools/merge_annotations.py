import os
import json
from collections import defaultdict

keywords = [
    'alternative', 'concession', 'condition', 'contrast', 'disjunctive',
    'location', 'precedence', 'reason', 'result', 'succession'
]
folders = [
    '../../data/build/disc_nli/test_annotations/ann_lingnli',
    '../../data/build/disc_nli/test_annotations/ann_mnli',
    '../../data/build/disc_nli/test_annotations/ann_wanli'
]

content_dict = {key: [] for key in keywords}
uid_counts = defaultdict(int)
lingnli_uid = 0

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
                    if each_conn in keywords:
                        content_dict[each_conn].append(json_obj)


for keyword, json_objects in content_dict.items():
    output_filename = f"../../data/build/disc_nli/test_annotations/merged/{keyword}_merged.jsonl"
    output_file_with_conn = f"../../data/build/disc_nli/test_annotations/generated/{keyword}_gen.jsonl"
    output_file_without_conn = f"../../data/build/disc_nli/test_annotations/original/{keyword}_original.jsonl"
    sorted_objects = sorted(json_objects, key=lambda x: x['uid'])

    with open(output_filename, 'w', encoding='utf-8-sig') as outfile, \
         open(output_file_with_conn, 'w', encoding='utf-8-sig') as outfile_with, \
         open(output_file_without_conn, 'w', encoding='utf-8-sig') as outfile_without:
        for json_obj in json_objects:
            json_str = json.dumps(json_obj, ensure_ascii=False)
            outfile.write(json_str + '\n')
            if 'modified_conn' in json_obj:
                outfile_with.write(json_str + '\n')
            else:
                outfile_without.write(json_str + '\n')

print("JSONL files have been successfully merged and processed based on the keywords.")
