import json
from tqdm import tqdm


def save_jsonl(d_list, filename):
    print("Save to Jsonl:", filename)
    with open(filename, encoding='utf-8-sig', mode='w') as out_f:
        for item in d_list:
            out_f.write(json.dumps(item) + '\n')


def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8-sig', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in tqdm(in_f):
            item = json.loads(line.strip())
            d_list.append(item)
    return d_list


wanli_train = load_jsonl("../../data/wanli/train.jsonl")
train_data_list = []
for each_line in wanli_train:
    train_data_list.append({"uid": str(each_line["id"]) + "wanli" + str(each_line["pairID"]),
                            "label": each_line["gold"][0], "premise": each_line["premise"],
                            "hypothesis": each_line["hypothesis"]})
save_jsonl(train_data_list, "../../data/build/wanli/train.jsonl")

wanli_test = load_jsonl("../../data/wanli/test.jsonl")
test_data_list = []
for each_line in wanli_test:
    test_data_list.append({"uid": str(each_line["id"]) + "wanli" + str(each_line["pairID"]),
                           "label": each_line["gold"][0], "premise": each_line["premise"],
                           "hypothesis": each_line["hypothesis"]})
save_jsonl(test_data_list, "../../data/build/wanli/test.jsonl")
