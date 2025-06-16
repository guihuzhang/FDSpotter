import json
import re
import random
from tqdm import tqdm
from copy import deepcopy
import spacy
nlp = spacy.load("en_core_web_sm")


adverbial_group = {
    "result": ["consequently", "hence", "thereby", "therefore", "thus", "as a consequence", "as a result", ],
    "contrast": ["however", "conversely", "on the contrary", "on the other hand", "by contrast", "in contrast", "by comparison", "in comparison", ],
    "alternative": ["alternatively", "as an alternative", "instead", ],
    "synchronous": ["at the same time", "meanwhile", "simultaneously", "meantime"],
    "succession": ["earlier", "previously", ],
    "precedence": ["subsequently", "afterwards", "afterward", "thereafter", "later", "then", ],  #
    "similarity": ["likewise", "in the same way", "similarly", "in a similar way"],
}

preposition_group = {
    "concession": ["in spite of", "despite", "albeit", "regardless of"],
    "reason": ["because of", "due to", "as a result of", "as a consequence of", ],
    "result": ["so that", "with the purpose of", "for the purpose of", "in order to"],  # ,
    "contrast": ["rather than"],
    "alternative": ["instead of", ],
    "condition": ["depending on", "depending upon"],
    "synchronous": ["during"],
}

conjunction_or_relative = {
    "concession": ["although", "even though", "though", "even if", ],
    "reason": ["because"],
    "contrast": ["whereas", "but"],
    "disjunctive": ["unless", ],
    "condition": ["if", "provided that", "in case", "depending on", "depending upon"],
    "synchronous": ["when"],
    "precedence": ["before", ],
    "succession": ["after", ],
    "location": ["where", ],
    "separation": ["away from where", "far from where", "not near where", ]
}

conflicting_groups = {
    "concession": ["reason", "result", "condition", ],
    "reason": ["result", "precedence", "disjunctive", "concession"],
    "result": ["reason", "succession", "condition", "contrast"],
    "contrast": ["similarity", "result", "reason"],
    "disjunctive": ["condition", "reason", "result"],
    "alternative": ["reason", "similarity"],
    "condition": ["result", "disjunctive", "concession"],
    "precedence": ["succession", "reason"],  #
    "succession": ["precedence", "result"],  #
    "location": ["separation"],
    "similarity": ["contrast", "alternative"]
}

useless_discourse = ["for example", "for instance", "and", "in addition", "plus", "as well as", "in other words", "in particular", "especially", "specifically", "in short", "in sum", "to sum up", "briefly", "in summary", "overall", "particularly", "precisely", "with the purpose of", "for the purpose of", ]


def check_valid_conn(tmp_txt, tmp_conn):
    doc = nlp(tmp_txt)
    matcher = spacy.matcher.Matcher(nlp.vocab)
    tmp_pattern = [{'LOWER': word} for word
                   in tmp_conn.split()]
    matcher.add("connective", [tmp_pattern])
    matches = matcher(doc)
    for match_id, start, end in matches:
        has_verb_before = any(t.pos_ in ["VERB", "AUX"] for t in doc[:start])
        has_verb_after = any(t.pos_ in ["VERB", "AUX"] for t in doc[end:])
        if has_verb_after and has_verb_before:
            return True
    return False


output_file = open("../../data/build/lingnli/train_construct.jsonl", "w", encoding="utf-8-sig")

with open("../../data/build/lingnli/train.jsonl", "r", encoding="utf-8-sig") as file:
    for line in tqdm(file):
        data = json.loads(line)
        hypothesis = data["hypothesis"].lower()
        found_groups = set()
        for group in conflicting_groups:
            if group in adverbial_group:
                adv_keys = adverbial_group[group]
                for keyword in adv_keys:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    if pattern.search(hypothesis) and check_valid_conn(hypothesis, keyword):
                        found_groups.add(group)
                        # print(keyword)
                        # print(hypothesis)
                        # print(check_valid_conn(hypothesis, keyword))
                        if (keyword == "as a consequence" and "as a consequence of" in hypothesis) or (
                                keyword == "as a result" and "as a result of" in hypothesis):
                            pass
                        else:
                            choosing_group = [cg for cg in conflicting_groups[group]
                                              if cg in adverbial_group]
                            if data["label"] == "e" and choosing_group:
                                tmp_data = deepcopy(data)
                                # chosen_conflict_group = random.choice(choosing_group)
                                for chosen_conflict_group in choosing_group:
                                    replacing_words = random.choice(adverbial_group[chosen_conflict_group])
                                    tmp_data["hypothesis"] = pattern.sub(replacing_words, hypothesis, count=0)
                                    tmp_data["original_conn"] = [group]
                                    tmp_data["modified_conn"] = [chosen_conflict_group]
                                    tmp_data["replaced"] = [keyword, replacing_words]
                                    tmp_data["label"] = "c"
                                    json.dump(tmp_data, output_file)
                                    output_file.write("\n")
                                    print(tmp_data)
            if group in conjunction_or_relative:
                cr_keys = conjunction_or_relative[group]
                for keyword in cr_keys:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    if pattern.search(hypothesis):
                        found_groups.add(group)
                        if (keyword == "if" and "even if" in hypothesis) or (keyword == "because" and "because of" in hypothesis) or (keyword == "though" and "even though" in hypothesis) or (keyword == "instead" and "instead of" in hypothesis):
                            pass
                        else:
                            choosing_group = [cg for cg in conflicting_groups[group]
                                              if cg in conjunction_or_relative]
                            if data["label"] == "e" and choosing_group:
                                tmp_data = deepcopy(data)
                                # chosen_conflict_group = random.choice(choosing_group)
                                for chosen_conflict_group in choosing_group:
                                    replacing_words = random.choice(conjunction_or_relative[chosen_conflict_group])
                                    tmp_data["hypothesis"] = pattern.sub(replacing_words, hypothesis, count=0)
                                    tmp_data["original_conn"] = [group]
                                    tmp_data["modified_conn"] = [chosen_conflict_group]
                                    tmp_data["replaced"] = [keyword, replacing_words]
                                    tmp_data["label"] = "c"
                                    json.dump(tmp_data, output_file)
                                    output_file.write("\n")
            if group in preposition_group:
                pr_keys = preposition_group[group]
                for keyword in pr_keys:
                    pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                    if pattern.search(hypothesis):
                        found_groups.add(group)
                        choosing_group = [cg for cg in conflicting_groups[group]
                                          if cg in preposition_group]
                        if data["label"] == "e" and choosing_group:
                            tmp_data = deepcopy(data)
                            # chosen_conflict_group = random.choice(choosing_group)
                            for chosen_conflict_group in choosing_group:
                                replacing_words = random.choice(preposition_group[chosen_conflict_group])
                                tmp_data["hypothesis"] = pattern.sub(replacing_words, hypothesis, count=0)
                                tmp_data["original_conn"] = [group]
                                tmp_data["modified_conn"] = [chosen_conflict_group]
                                tmp_data["replaced"] = [keyword, replacing_words]
                                tmp_data["label"] = "c"
                                json.dump(tmp_data, output_file)
                                output_file.write("\n")
        if found_groups:
            data['original_conn'] = list(found_groups)
            json.dump(data, output_file)
            output_file.write("\n")

output_file.close()
