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
    "similarity": ["contrast", "alternative"],
    "synchronous": ["precedence", "succession"]
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


collected_data = []
premise_counts = {}

with open("../../data/build/lingnli/train.jsonl", "r", encoding="utf-8-sig") as file:
    for line in tqdm(file):
        data = json.loads(line)
        hypothesis = data["hypothesis"].lower()
        premise = data["premise"].lower()
        found_keys = set()
        if data["label"] == "e":
            for group in conflicting_groups:
                if group in adverbial_group:
                    adv_keys = adverbial_group[group]
                    for keyword in adv_keys:
                        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                        if pattern.search(hypothesis) and check_valid_conn(hypothesis, keyword):
                            found_keys.add(group)
                            if (keyword == "as a consequence" and "as a consequence of" in hypothesis) or (
                                    keyword == "as a result" and "as a result of" in hypothesis):
                                pass
                            else:
                                choosing_group = [cg for cg in conflicting_groups[group]
                                                  if cg in adverbial_group]
                                if data["label"] == "e" and choosing_group:
                                    # chosen_conflict_group = random.choice(choosing_group)
                                    for chosen_conflict_group in choosing_group:
                                        tmp_data_adv = deepcopy(data)
                                        replacing_words = random.choice(adverbial_group[chosen_conflict_group])
                                        tmp_data_adv["hypothesis"] = pattern.sub(replacing_words, hypothesis, count=0)
                                        tmp_data_adv["original_conn"] = [group]
                                        tmp_data_adv["modified_conn"] = [chosen_conflict_group]
                                        tmp_data_adv["replaced"] = [keyword, replacing_words]
                                        tmp_data_adv["label"] = "c"
                                        collected_data.append(tmp_data_adv)
                if group in conjunction_or_relative:
                    cr_keys = conjunction_or_relative[group]
                    for keyword in cr_keys:
                        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                        if pattern.search(hypothesis):
                            found_keys.add(group)
                            if (keyword == "if" and "even if" in hypothesis) or (
                                    keyword == "because" and "because of" in hypothesis) or (
                                    keyword == "though" and "even though" in hypothesis) or (
                                    keyword == "instead" and "instead of" in hypothesis):
                                pass
                            else:
                                choosing_group = [cg for cg in conflicting_groups[group]
                                                  if cg in conjunction_or_relative]
                                if data["label"] == "e" and choosing_group:
                                    # chosen_conflict_group = random.choice(choosing_group)
                                    for chosen_conflict_group in choosing_group:
                                        tmp_data_con = deepcopy(data)
                                        replacing_words = random.choice(conjunction_or_relative[chosen_conflict_group])
                                        tmp_data_con["hypothesis"] = pattern.sub(replacing_words, hypothesis, count=0)
                                        tmp_data_con["original_conn"] = [group]
                                        tmp_data_con["modified_conn"] = [chosen_conflict_group]
                                        tmp_data_con["replaced"] = [keyword, replacing_words]
                                        tmp_data_con["label"] = "c"
                                        collected_data.append(tmp_data_con)
                if group in preposition_group:
                    pr_keys = preposition_group[group]
                    for keyword in pr_keys:
                        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                        if pattern.search(hypothesis):
                            found_keys.add(group)
                            choosing_group = [cg for cg in conflicting_groups[group]
                                              if cg in preposition_group]
                            if data["label"] == "e" and choosing_group:
                                # chosen_conflict_group = random.choice(choosing_group)
                                for chosen_conflict_group in choosing_group:
                                    tmp_data_prep = deepcopy(data)
                                    replacing_words = random.choice(preposition_group[chosen_conflict_group])
                                    tmp_data_prep["hypothesis"] = pattern.sub(replacing_words, hypothesis, count=0)
                                    tmp_data_prep["original_conn"] = [group]
                                    tmp_data_prep["modified_conn"] = [chosen_conflict_group]
                                    tmp_data_prep["replaced"] = [keyword, replacing_words]
                                    tmp_data_prep["label"] = "c"
                                    collected_data.append(tmp_data_prep)
        if found_keys:
            data['original_conn'] = list(found_keys)
            collected_data.append(data)
            if premise in premise_counts:
                premise_counts[premise] += 1
            else:
                premise_counts[premise] = 1

sorted_data = sorted(collected_data, key=lambda x: x['premise'])
file_handles = {}
for entry in sorted_data:
    for original_conn in entry["original_conn"]:
        file_name = f"../../data/build/lingnli/train2split_{original_conn.replace('/', '_')}.jsonl"
        if original_conn == "result":
            print(entry)
        if original_conn not in file_handles:
            file_handles[original_conn] = open(file_name, 'w', encoding='utf-8')
        file_handles[original_conn].write(json.dumps(entry) + '\n')

print("Total unique premises:", len(premise_counts))
