import os
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset
from pynvml import *
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

# # wandb seeting
os.environ["WANDB_API_KEY"] = "d2cc1ce195e31ad61c8e2127df2d4369adf4b493"
os.environ["WANDB_PROJECT"] = "Ti_Bert"
os.environ["WANDB_MODE"] = "offline"


SEED = 666
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

parser = ArgumentParser()
parser.add_argument("--cache_dir", default=None, type=str)
parser.add_argument("--save_dir", required=True, type=str)
args = parser.parse_args()

# add saved dir
for p in ["tokenizer_saved", "tokens_saved", "model_saved", "model_final_saved"]:
    locals()[p + "_dir"] = Path(args.save_dir) / p

tokenizer = AutoTokenizer.from_pretrained(tokenizer_saved_dir)
config = AutoConfig.from_pretrained(
    "/hy-tmp/deberta-v3-xsmall/models--microsoft--deberta-v3-xsmall", cache_dir=args.cache_dir
)


print(f"model config: \n {config}")

# 使用配置初始化DebertaV2ForMaskedLM模型
model = DebertaV2ForMaskedLM(config=config)

model_size = sum(t.numel() for t in model.parameters())
print(f"Pretrained model: {model_size/1000**2:.1f}M parameters")

config.save_pretrained("/path/to/save/configuration")

model.resize_token_embeddings(len(tokenizer))

# load tokens
dataset_train = Dataset.load_from_disk(Path(tokens_saved_dir) / "train")
dataset_val = Dataset.load_from_disk(Path(tokens_saved_dir) / "val")
print(dataset_train, dataset_val)



dataset_train.set_format(type="torch", columns=["input_ids"])
dataset_val.set_format(type="torch", columns=["input_ids"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    print("*" * 100)
    print(type(preds), preds.shape, preds)
    print(type(labels), labels.shape, labels)

    return {"perplexity": 1, "calculated_loss": 1}


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


# print_gpu_utilization()

training_args = TrainingArguments(
    output_dir=model_saved_dir,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,  # 8,
    per_device_eval_batch_size=40,  # 80,
    gradient_accumulation_steps=72,
    learning_rate=1e-4,
    weight_decay=1e-2,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    max_grad_norm=1.0,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_ratio=0.048,
    warmup_steps=10_000,
    eval_steps=10,
    save_strategy='steps',
    logging_strategy='steps',
    logging_steps=100,
    save_steps=500,
    save_total_limit=10,
    seed=SEED,
    data_seed=SEED,
    fp16=True,
    optim='adamw_torch',
    max_steps=130000,

)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
)

# 检查是否存在检查点，如果有，则从最新的检查点恢复训练
resume = None if len(os.listdir(model_saved_dir)) == 0 else True
train_res = trainer.train(resume_from_checkpoint='/hy-tmp/TiBert_data/output/model_saved/checkpoint-22000')
print(train_res)

# 保存模型
trainer.save_model(model_final_saved_dir)

# 可选：评估模型性能
train_output = trainer.evaluate(dataset_train)
eval_output = trainer.evaluate(dataset_val)  # 确保传入验证数据集
print(train_output)
print(eval_output)

print("Script run COMPLETE!!!")

