#!/bin/sh

cache_dir=./cache
save_dir=/hy-tmp/TiBert_data/output
train_corpus_file=/hy-tmp/TiBert_data/train_corpus.txt
val_corpus_file=/hy-tmp/TiBert_data/eval_corpus.txt
train_norm_file=/hy-tmp/TiBert_data/train_corpus_norm.txt
val_norm_file=/hy-tmp/TiBert_data/val_corpus_norm.txt

python -u Ti_json_combination.py

python -u /hy-tmp/TiBert/pretrain/Ti_corpus_normalize.py \
   --train_corpus_file $train_corpus_file \
   --val_corpus_file $val_corpus_file \
   --train_norm_file $train_norm_file \
   --val_norm_file $val_norm_file


python -u tokenizer_train.py \
    --train_norm_file $train_norm_file \
    --val_norm_file $val_norm_file \
    --save_dir $save_dir \
    --cache_dir $cache_dir

python -u tokens_count.py \
    --train_norm_file $train_norm_file \
    --val_norm_file $val_norm_file \
    --save_dir $save_dir

python -u model_train.py \
    --save_dir $save_dir \
    --cache_dir $cache_dir