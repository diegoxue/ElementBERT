[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
# Universal Semantic Embeddings for Elements
ElementBERT and Down-stream Tasks

This repository is the code inplementaion of the paper: "Universal Semantic Embeddings of Chemical Elements for Enhanced Materials Inference and Discovery".
+ The pre-training of ElementBERT is shown in file "ElementBERT_training"
+ The down-stream tasks code is shown in file "downstream_tasks"
+ The requirements files are listed separately in the above two files. We use different python envirnments to do these two studies, as the pre-training of ElementBERT is seperate from materails inferences.

## Setup
The code we provide is friendly to rookies. Just simply download all files to get started!

## Instructions
+ We've tested on various platforms, including `Windows` and `Linux`, as well as different python versions, feel free to run tests on `python 3.10` to `python 3.12` environments. You can create your envirnment using `conda` or `pip`.
+ We recommend using [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). Then activate the enviroment for pre-training or down-stream tasks, like `conda activate trainingBERT`or `conda activate gprtests`.
+ Make sure to run `pip install --ignore-installed -r requirements.txt` to install the required packages. Note: If you failed some installations in this step, probably pytorch, you can install the packages separately using `pip install [package_name]` or `conda install [package_name]`.
+ Then you are ALL SET!

### Pre-training data harvest
If you have the access to database like Web of Science or Scopus, you can easily download the abstracts you want to pre-train on. The abstracts are well organized in the downloaded .xlsx or .csv files, you can easily extract these text to a .md or .txt file using a simple python code (which we didn't provide :)).

### Pre-training steps
Training LLMs requires huge amount of computing resorces, so that we submitted our training job on a cluster server using the Slurm job scheduling system. The `.sh` file for submitting the job is provided in the repo.
The GPU memory usage of this model is about 30 GB. So we suggest using A100 40G to pre-train the model.
#### File explaination
+ `Alloy_run.sh` is the submitting file we use. It schedully runs 4 python scripts for data processing, tokenizer training and model training.
+ `Alloy_json_combination.py` is excecuted 1st, to split a corpus of text data into training and evaluation (validation) sets.
+ `Alloy_corpus_normalize.py` is excecuted 2nd, to processes text data using BERT-style normalization.
+ `Alloy_tokenizer_train.py` is excecuted 3rd, to create and train a custom tokenizer based on `DeBERTa-v3-xsmall` model.
+ `Alloy_model_train.py` is excecuted last, to train the BERT model using `DebertaV2ForMaskedLM` with `transformers` python package.

### Down-stream tasks implementation


## License
Universal Semantic Embeddings for Elements is distributed under an MIT License.
