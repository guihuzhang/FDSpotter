# Downstream Task: AggreFact

This folder includes the codes for reproducing FactOverlap performance on AggreFact benchmark. 

To simply obtain the scores reported in Table 10, please run:

    python3 af_result_reproduce.py

If you want to obtain the results step by step, please run the following scripts.

**1.** For calculating intrinsic and extrinsic confidence, please run

    python3 step1_intrinsic_extrinsic_confidence.py

The input file "af_gpt4o_ext_prob.json" is the GPT4 output from OpenAI API, available at https://huggingface.co/datasets/Inria-CEDAR/FDSpotter/tree/main

In the output file "out1_intrinsic_extrinsic_confidence.json", "cls conf" field is intrinsic confidence, and "cls faith" field is extrinsic confidence. 

For using "neuralcoref" and "mosestokenizer" packages, the environment should be under python3.7 with spacy==2.1.0 . Otherwise there would be compatibility issues. 

**2.** To get the CSV file for final result computation, please run 

    python3 step2_summary_scoring_binary_log.py

and the output file is "out2_aggre_fact_overlap_gpt4o_plogp_binary.5.csv". I run this code under python 3.10 with latest Transformers library. 