# WebNLG Metric Correlation

The files in this folder should be used to compute correlation via bootstrapping between human judgements and automatic metrics.

In metrics_results folder we have already the precomputed metrics.

The FactOverlap correlation can be run as follows:

    python3 corr_sys_text_2020.py parameter_2020.json

To obtain the scores of FactOverlap, please run:

    python3 fact_overlap523.py

The GPT extraction outputs of the ground-truth and generated texts are in extraction_output folder. 