# FDSPotter Training Instructions

This is the code for training for atomic fact and discourse relation entailment model.

## Dataset format

The dataset files are all in JSONL format (one JSON per line). Below is one example with self-explanatory fields. The atomic fact or discourse relation to be checked is in "hypothesis", with elements delimited by "|" or ",". The premise is the text in which we check whether the hypothesis is explained in it. 

    {"uid": "lingnli1441a", "hypothesis": "Voters will play the role of budget analyst \| when \| the next recession arrives.", "premise": "But when the cushion is spent in a year or two, or when the next recession arrives, the disintermediating voters will find themselves playing the roles of budget analysts and tax wonks.", "label": "e", "original_conn": ["synchronous"], "source": "lingnli"}

"e" => "entail", "c" => "contradict", and "n" => neutral for labels in annotation.

## Model Training Command

The model training command is in 

    script/deberta_large_w2.sh

## Reference

This code is partially modified from the github repo
https://github.com/facebookresearch/anli/tree/main 