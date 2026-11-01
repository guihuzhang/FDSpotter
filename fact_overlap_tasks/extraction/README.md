# Extraction gold annotations

[`extraction_gold.csv`](extraction_gold.csv) contains reference atomic facts and discourse relations for 50 sentences from the Causal News Corpus, used in the FDSpotter extraction experiment.

## File format

The UTF-8 CSV has one row per sentence and the following columns:

| Column | Contents |
|---|---|
| `corpus` | Source corpus (`cnc`). |
| `doc_id` | Source document identifier. |
| `sent_id` | Source sentence identifier. |
| `sentence` | Input sentence. |
| `gold_facts` | Reference atomic facts in a Markdown table. |
| `gold_discourse` | Reference discourse relations in a Markdown table. |

Atomic facts have five fields: subject, predicate, direct object, indirect object, and short adverbial and complement. Discourse relations have three fields: fact1, connective, and fact2. A table containing only its heading and column headers has no annotated tuples.

## Reading the annotations

From this directory, using Python 3:

```python
import csv

with open("extraction_gold.csv", encoding="utf-8", newline="") as source:
    examples = list(csv.DictReader(source))

print(len(examples))
print(examples[0]["sentence"])
print(examples[0]["gold_facts"])
print(examples[0]["gold_discourse"])
```

This prints `50`, followed by the first sentence and its reference extractions.

## Source

Tan et al. (2022), [The Causal News Corpus: Annotating Causal Relations in Event Sentences from News](https://aclanthology.org/2022.lrec-1.246/), LREC 2022. The original corpus is available in the [CausalNewsCorpus repository](https://github.com/tanfiona/CausalNewsCorpus). The atomic-fact and discourse-relation extractions in this CSV are the FDSpotter annotations, rather than the original corpus labels.
