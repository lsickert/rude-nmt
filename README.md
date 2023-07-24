# rude-nmt

This repository contains the code used for my master thesis on examining formality in machine translation through interpretability methods at the [University of Groningen](https://www.rug.nl/).

The Jupyter notebooks contain the main experiments, while the different python scripts contain the functions required to run the experiments.

The shell scripts contain commands to run the compute-intensive tasks such as the full translation of the datasets or the feature attributions on the [HPC cluster](https://wiki.hpc.rug.nl/habrok/start) of the University of Groningen. Some of the notebooks assume that these tasks have been run beforehand and the respective augmented datasets are available.

## Installation

> A Python version of at least 3.8 is required to run the code.

All dependecies needed to run the module can be installed by running the following command:

```shell
pip3 install -r requirements.txt
```

Installation through [poetry](https://python-poetry.org/) is also possible.

## How To Run

All the scripts can be started and configured from the command line with the following options:

|Option|Description|Default Value|
|------|-----------|-------------|
|`data`|The dataset to use for the script (either Tatoeba or IWSLT2023)| None **(required)** |
|`src_lang`|The source language of the translation direction, one of either German `de` or Korean `ko`| None |
|`trg_lang`|The target language of the translation direction, one of either German `de` or Korean `ko`| None |
|`label_data`|Whether the data should be annotated with formality labels and POS tags|False|
|`force_regenerate`|Forces the regeneration of all steps of the script even if a cached version is available|False|
|`force_redownload`|Forces a redownload of the Tatoeba dataset (the IWSLT dataset is included in the repository)|False|
|`translate`|Run the translation scripts|False|
|`attribute`|Run the feature attribution scripts with the given attribution method (should be one of the [feature attribution methods](https://inseq.readthedocs.io/en/latest/main_classes/feature_attribution.html) supported by the InSeq package)|None|
|`use_ds`|Use a custom dataset for the script instead of one of the standard ones created by earlier steps in the process |None|
|`save_csv`|Save the dataset as a csv-file after processing|False|

An example run that would label and translate the Tatoeba dataset in the German --> Korean direction could look like this: `python -u main.py --data tatoeba --translate --src_lang de --trg_lang ko --label_data`
