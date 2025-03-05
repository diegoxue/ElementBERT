#!/bin/bash

#SBATCH --job-name=ElementBERT       
#SBATCH --gres=gpu:1    
#SBATCH --output=train_%j.log    
#SBATCH --error=train_%j.err         

module load compilers/cuda/12.1
module load miniforge3/24.1
source activate transformer

# 设置基础配置
export CUDA_VISIBLE_DEVICES=0
cache_dir="./cache"
save_dir="./output"
train_corpus_file="./train_corpus.txt"
val_corpus_file="./eval_corpus.txt"
train_norm_file="./train_corpus_norm.txt"
val_norm_file="./val_corpus_norm.txt"

# 创建必要的目录
mkdir -p $cache_dir $save_dir

# 1. 运行数据组合脚本
echo "Starting data combination..."
python -u Alloy_json_combination.py
echo "Data combination completed"

# 2. 运行文本标准化脚本
echo "Starting text normalization..."
python -u Alloy_corpus_normalize.py \
    --train_corpus_file $train_corpus_file \
    --val_corpus_file $val_corpus_file \
    --train_norm_file $train_norm_file \
    --val_norm_file $val_norm_file
echo "Text normalization completed"

# 3. 运行tokenizer训练脚本
echo "Starting tokenizer training..."
python -u Alloy_tokenizer_train.py \
    --train_norm_file $train_norm_file \
    --val_norm_file $val_norm_file \
    --save_dir $save_dir \
    --cache_dir $cache_dir
echo "Tokenizer training completed"

# 4. 运行模型训练脚本
echo "Starting model training..."
python -u Alloy_model_train.py \
    --save_dir $save_dir \
    --cache_dir $cache_dir
echo "Model training completed"

echo "All training steps completed successfully!"