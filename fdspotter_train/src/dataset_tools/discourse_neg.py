import json
import re
from tqdm import tqdm

discourse_group_new = {
    "concession": ["although", "in spite of", "even though", "though", "despite", "even if", "albeit", "regardless"],

    "reason": ["because", "due to", "as a result of", "as a consequence of", "because of"],

    "result": ["consequently", "hence", "thereby", "therefore", "thus", "so", "so that", "for the purpose of", " in order to", "with the purpose of", "as a result", "as a consequence", "for this reason", "for that reason"],

    "contrast": ["by contrast", "conversely", "however", "in contrast", "on the contrary", "on the other hand", "rather than", "whereas", "by comparison", "in comparison", "but"],

    "disjunctive": ["otherwise", "unless", "except"],

    "alternative": ["alternatively", "as an alternative", "instead", ],

    "condition": ["if", "provided that", "in case", "depending on", "depending upon"],

    "synchronous": ["when", "during", "at the same time", "meanwhile", "simultaneously", "meantime"],

    "precedence": ["earlier", "previously", "until", "till", "before"],

    "succession": ["subsequently", "afterwards", "afterward", "thereafter", "then", "after", "later"],

    "similarity": ["likewise", "in the same way", "similarly", "in a similar way"],

    "location": ["where"],
}

conjunction_or_relative = {
    "concession": ["although", "in spite of", "even though", "though", "despite", "even if", "albeit", "regardless of"],
    "reason": ["because of", "because", "due to", "as a result of", "as a consequence of", ],
    "result": ["so that", "with the purpose of", "for the purpose of", ],  # "in order to",
    "contrast": ["rather than", "whereas", "but"],
    "disjunctive": ["unless", ],
    "alternative": ["instead of", ],
    "condition": ["if", "provided that", "in case", "depending on", "depending upon"],
    "synchronous": ["when", "during"],
    "precedence": ["until", "till", "before"],
    "succession": ["after", ],
    "location": ["where", ],
    "separation": ["away from where", "far from where", "not near where", "outside of where", ]
}
adverbial_group = {
    "result": ["consequently", "hence", "thereby", "therefore", "thus", "as a consequence", ],
    "contrast": ["however", "conversely", "on the contrary", "on the other hand", "by contrast", "in contrast", "by comparison", "in comparison", ],
    "disjunctive": ["otherwise", ],
    "alternative": ["alternatively", "as an alternative", "instead", ],
    "synchronous": ["at the same time", "meanwhile", "simultaneously", "meantime"],
    "precedence": ["earlier", "previously", ],
    "succession": ["subsequently", "afterwards", "afterward", "thereafter", "then", "later"],
    "similarity": ["likewise", "in the same way", "similarly", "in a similar way"],
}

useless_discourse = ["for example", "for instance", "and", "in addition", "plus", "as well as", "in other words", "in particular", "especially", "specifically", "in short", "in sum", "to sum up", "briefly", "in summary", "overall", "particularly", "precisely", ]


output_files = {
    key: open(f"../../data/analysis/anli3/{key}_output.jsonl", "w", encoding="utf-8-sig") for key in discourse_group_new
}

with open("../../data/build/anli/r3/train.jsonl", "r", encoding="utf-8-sig") as file:
    for line in tqdm(file):
        data = json.loads(line)
        hypothesis = data.get("hypothesis", "").lower()
        for group, keywords in discourse_group_new.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', hypothesis) for keyword in keywords):
                json.dump(data, output_files[group])
                output_files[group].write("\n")
for file in output_files.values():
    file.close()
