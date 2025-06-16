import json
import re
from tqdm import tqdm

discourse_group = {
    "concession": ["although", "in spite of", "even though", "though", "despite", "even if", "albeit", "regardless"],
    "reason": ["because", "due to", "as a result of", "as a consequence of", "because of"],
    "result": ["consequently", "hence", "thereby", "therefore", "thus", "so", "for the purpose of", " in order to", "with the purpose of"],
    "contrast": ["by contrast", "conversely", "however", "in contrast", "instead", "on the contrary", "on the other hand", "rather than", "whereas", "by comparison", "in comparison", "whereas"],
    "similarity": ["likewise", "in the same way", "similarly", "in a similar way"],
    "exception": ["otherwise", "alternatively", "as an alternative", "unless"],
    "condition": ["if", "provided that", "in case", "depending on", "depending upon"],
    "location": ["where"],
    "synchronous": ["when", "during", "at the same time", "meanwhile", "simultaneously", "meantime"],
    "precedence": ["subsequently", "afterwards", "afterward", "thereafter", "then"],
    "succession": ["after", ]
}

discourse_group_new = {
    "concession": ["although", "in spite of", "even though", "though", "despite", "even if", "albeit", "regardless"],

    "reason": ["because", "due to", "as a result of", "as a consequence of", "because of"],

    "result": ["consequently", "hence", "thereby", "therefore", "thus", "so that", "for the purpose of", "in order to", "with the purpose of", "as a result", "as a consequence", "for this reason", "for that reason"],

    "contrast": ["by contrast", "conversely", "however", "in contrast", "on the contrary", "on the other hand", "rather than", "by comparison", "in comparison", "but"],

    "disjunctive": ["otherwise", "unless", "except"],

    "alternative": ["alternatively", "as an alternative", "instead", ],

    "condition": ["if", "provided that", "in case", "depending on", "depending upon"],

    "synchronous": ["when", "during", "at the same time", "meanwhile", "simultaneously", "meantime"],

    "precedence": ["earlier", "previously", "until", "till", "before"],

    "succession": ["subsequently", "afterwards", "afterward", "thereafter", "then", "after", "later"],

    "similarity": ["likewise", "in the same way", "similarly", "in a similar way"],

    "location": ["where"],
}

useless_group = {
    "conjunction": ["additionally", "also", "and", "as well", "as well as", "further", "furthermore", "in fact", "in addition", "plus", "moreover", "whereas", ],

    "specification": ["in particular", "especially", "specifically", "particularly", ],

    "restatement": ["in other words"],

    "instantiation": ["for example", "for instance", ],

    "generalisation": ["in short", "in sum", "to sum up", "briefly", "in summary", "overall", ],

    "multiple": ["as", "as if", "as though", "as long as", "as soon as", "insofar as", "besides", "either", "else", "finally", "for", "indeed", "in turn", "lest", "much as", "neither", "nevertheless", "next", "nor", "once", "or", "separately", "still", "ultimately", "yet", "so", "otherwise", "except"],
}

useless_discourse = ["precisely"]

output_file = open("../../data/build/wanli/discourse_test.jsonl", "w", encoding="utf-8-sig")

with open("../../data/build/wanli/test.jsonl", "r", encoding="utf-8-sig") as file:
    for line in tqdm(file):
        data = json.loads(line)
        hypothesis = data.get("hypothesis", "").lower()
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', hypothesis) for keywords in discourse_group.values() for keyword in keywords):
            json.dump(data, output_file)
            output_file.write("\n")

output_file.close()
