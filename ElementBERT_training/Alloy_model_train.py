import logging
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DebertaV2ForMaskedLM,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


@dataclass
class ModelConfig:
    """模型训练配置类"""

    cache_dir: Optional[str] = None
    save_dir: str = "output"
    seed: int = 666
    batch_size: int = 8
    eval_batch_size: int = 40
    gradient_accumulation_steps: int = 72
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    num_epochs: int = 1
    warmup_ratio: float = 0.048
    warmup_steps: int = 10_000
    max_steps: int = 130_000
    save_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 10


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, config: ModelConfig):
        """
        初始化模型训练器

        Args:
            config: 模型配置对象
        """
        self.config = config
        self.setup_environment()
        self.setup_paths()
        self.setup_logging()

    def setup_environment(self) -> None:
        """设置环境变量和随机种子"""
        os.environ.update(
            {
                "WANDB_API_KEY": "d2cc1ce195e31ad61c8e2127df2d4369adf4b493",
                "WANDB_PROJECT": "Ti_Bert",
                "WANDB_MODE": "offline",
            }
        )
        set_seed(self.config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Training on {self.device}")

    def setup_paths(self) -> None:
        """创建必要的目录结构"""
        self.paths = {}
        for path_type in ["tokenizer", "tokens", "model", "model_final"]:
            dir_name = f"{path_type}_saved"
            path = Path(self.config.save_dir) / dir_name
            path.mkdir(parents=True, exist_ok=True)
            self.paths[dir_name] = str(path)

    def setup_logging(self) -> None:
        """配置日志系统"""
        log_dir = Path(self.config.save_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self._log_system_info()

    def _log_system_info(self) -> None:
        """记录系统信息"""
        logging.info("=" * 80)
        logging.info("Training started")
        if torch.cuda.is_available():
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("No GPU available, using CPU")
        logging.info("=" * 80)

    def load_model_and_tokenizer(
        self,
    ) -> Tuple[DebertaV2ForMaskedLM, PreTrainedTokenizer]:
        """加载模型和分词器"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.paths["tokenizer_saved"])
            config = AutoConfig.from_pretrained(
                "./deberta-v3-xsmall/models--microsoft--deberta-v3-xsmall",
                cache_dir=self.config.cache_dir,
            )
            logging.info(f"Model configuration:\n{config}")

            model = DebertaV2ForMaskedLM(config=config)
            model.resize_token_embeddings(len(tokenizer))

            model_size = sum(t.numel() for t in model.parameters())
            logging.info(
                f"Pretrained model size: {model_size / 1000**2:.1f}M parameters"
            )

            return model, tokenizer
        except Exception as e:
            logging.error(f"Error loading model and tokenizer: {str(e)}")
            raise

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """加载训练和验证数据集"""
        try:
            dataset_train = Dataset.load_from_disk(
                Path(self.paths["tokens_saved"]) / "train"
            )
            dataset_val = Dataset.load_from_disk(
                Path(self.paths["tokens_saved"]) / "val"
            )

            for dataset, name in [
                (dataset_train, "Training"),
                (dataset_val, "Validation"),
            ]:
                dataset.set_format(type="torch", columns=["input_ids"])
                logging.info(f"{name} dataset: {dataset}")

            return dataset_train, dataset_val
        except Exception as e:
            logging.error(f"Error loading datasets: {str(e)}")
            raise

    def get_training_args(self) -> TrainingArguments:
        """获取训练参数配置"""
        return TrainingArguments(
            output_dir=self.paths["model_saved"],
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-6,
            max_grad_norm=1.0,
            num_train_epochs=self.config.num_epochs,
            lr_scheduler_type="linear",
            warmup_ratio=self.config.warmup_ratio,
            warmup_steps=self.config.warmup_steps,
            eval_steps=10,
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            seed=self.config.seed,
            data_seed=self.config.seed,
            fp16=True,
            optim="adamw_torch",
            max_steps=self.config.max_steps,
        )

    def train(self) -> None:
        """执行模型训练流程"""
        try:
            model, tokenizer = self.load_model_and_tokenizer()
            dataset_train, dataset_val = self.load_datasets()

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )

            trainer = Trainer(
                model=model,
                args=self.get_training_args(),
                data_collator=data_collator,
                train_dataset=dataset_train,
                eval_dataset=dataset_val,
                tokenizer=tokenizer,
            )

            self._handle_training(trainer)
            self._save_and_evaluate(trainer, dataset_train, dataset_val)

        except Exception as e:
            logging.error(f"Training failed: {str(e)}", exc_info=True)
            raise

    def _handle_training(self, trainer: Trainer) -> None:
        """处理训练过程"""
        checkpoint_files = os.listdir(self.paths["model_saved"])
        if not checkpoint_files:
            logging.info("Starting training from scratch")
            train_result = trainer.train()
        else:
            latest_checkpoint = max(
                [f for f in checkpoint_files if f.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]),
            )
            checkpoint_path = str(Path(self.paths["model_saved"]) / latest_checkpoint)
            logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
            train_result = trainer.train(resume_from_checkpoint=checkpoint_path)

        logging.info(f"Training results:\n{train_result}")

    def _save_and_evaluate(
        self, trainer: Trainer, dataset_train: Dataset, dataset_val: Dataset
    ) -> None:
        """保存模型并进行评估"""
        trainer.save_model(self.paths["model_final_saved"])
        logging.info(f"Model saved to {self.paths['model_final_saved']}")

        train_metrics = trainer.evaluate(dataset_train)
        eval_metrics = trainer.evaluate(dataset_val)
        logging.info(f"Training evaluation results:\n{train_metrics}")
        logging.info(f"Validation evaluation results:\n{eval_metrics}")


def parse_args() -> ModelConfig:
    """解析命令行参数"""
    parser = ArgumentParser(description="模型训练脚本")
    parser.add_argument("--cache_dir", default=None, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    args = parser.parse_args()
    return ModelConfig(cache_dir=args.cache_dir, save_dir=args.save_dir)


def main() -> None:
    """主函数"""
    try:
        config = parse_args()
        trainer = ModelTrainer(config)
        trainer.train()
        logging.info("Training completed successfully!")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise
    finally:
        print("Script execution completed")


if __name__ == "__main__":
    main()
