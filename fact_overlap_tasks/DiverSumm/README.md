
# Downstream Task: DiverSumm

This folder includes the codes for reproducing the results on DiverSumm. If you want to quickly obtain the result reported in the paper, you can run the following code:

    cd fact_overlap
    python3 roc_auc_calc_p.py

If you want to obtain the result step by step, please read the following descriptions.

In code_extraction folder, you can find the code of generating .jsonl files for OpenAI batch submission (https://platform.openai.com/docs/guides/batch) in batch_file_make.py. The processed extractions can be found in .json files in the folder. 

In fact_overlap folder, it contains the code for obtaining the intrinsic confidence and extrinsic confidence for computing the final score. It is easy to modify the input and output file name to obtain the confidence of another split. 

    cd fact_overlap
    python3 fact_overlap4o256part3.py

In the output .json file, "cls conf" field is intrinsic confidence, and "cls faith" field is extrinsic confidence. Then the following script combines the .json outputs to obtain the final score. 
    
    python3 roc_auc_calc_p.py

Similar to AggreFact, when using "neuralcoref" and "mosestokenizer" packages, the environment should be under python3.7 with spacy==2.1.0 . Otherwise there would be compatibility issues. 

The folder data_input has original DiverSumm data. You can reproduce scores of INFUSE from the paper [Fine-Grained Natural Language Inference Based Faithfulness Evaluation for Diverse Summarisation Tasks](https://aclanthology.org/2024.eacl-long.102/) (Zhang et al., EACL 2024) with the following script: 

    cd data_input
    python3 infuse_from_csv.py --csv_input DiverSumm.csv --csv_output DiverSumm1.5.csv
    python3 roc_auc_calc_infuse.py

Please be careful that only with stanza==1.5 we can reproduce the result reported in the paper.   


