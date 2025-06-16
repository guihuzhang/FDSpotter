from datasets import load_dataset
from tqdm import tqdm
import json

dataset = load_dataset('tasksource/lingnli')

data_train = dataset["train"]
data_val = dataset["validation"]

val_dict = {}
output_dir = "../../data/build/lingnli/dev.jsonl"
tmp_dict = data_val.to_dict()
for idx in tqdm(range(data_val.num_rows)):
    tmp_key = tmp_dict["hypothesis"][idx] + " " + tmp_dict["premise"][idx] + " " + tmp_dict["label"][idx]
    if tmp_key not in val_dict:
        val_dict[tmp_key] = {"uid": output_dir + str(idx), "hypothesis": tmp_dict["hypothesis"][idx],
                             "premise": tmp_dict["premise"][idx], "label": tmp_dict["label"][idx][0]}
        print(tmp_key)
with open(output_dir, encoding='utf-8-sig', mode='w') as out_f:
    for each_key in val_dict:
        out_f.write(json.dumps(val_dict[each_key]) + '\n')
print(len(val_dict))

train_dict = {}
output_dir = "../../data/build/lingnli/train.jsonl"
tmp_dict = data_train.to_dict()
for idx in tqdm(range(data_train.num_rows)):
    tmp_key = tmp_dict["hypothesis"][idx] + " " + tmp_dict["premise"][idx] + " " + tmp_dict["label"][idx]
    if tmp_key not in train_dict:
        train_dict[tmp_key] = {"uid": output_dir + str(idx), "hypothesis": tmp_dict["hypothesis"][idx],
                               "premise": tmp_dict["premise"][idx], "label": tmp_dict["label"][idx][0]}
        print(tmp_key)
with open(output_dir, encoding='utf-8-sig', mode='w') as out_f:
    for each_key in train_dict:
        out_f.write(json.dumps(train_dict[each_key]) + '\n')
print(len(train_dict))
