import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Tokenizer配置类"""

    max_seq_length: int = 512
    vocab_size: int = 128_100
    base_model_path: str = "./deberta-v3-xsmall/models--microsoft--deberta-v3-xsmall"
    seed: int = 666


class TokenizerTrainer:
    """Tokenizer训练和处理类"""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        set_seed(config.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = None
        self.special_tokens = {}

    def load_datasets(self, train_file: str, val_file: str) -> DatasetDict:
        """加载训练和验证数据集"""
        return load_dataset("text", data_files={"train": train_file, "val": val_file})

    def get_training_corpus(
        self, dataset: Dataset, batch_size: int = 10000
    ) -> Iterator[List[str]]:
        """生成训练语料迭代器"""
        return (
            dataset["train"][i : i + batch_size]["text"]
            for i in range(0, len(dataset["train"]), batch_size)
        )

    def train_tokenizer(self, training_corpus: Iterator[List[str]]) -> None:
        """训练新的tokenizer"""
        base_tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
        self.tokenizer = base_tokenizer.train_new_from_iterator(
            text_iterator=training_corpus, vocab_size=self.config.vocab_size
        )
        logger.info(f"Original tokenizer size: {len(base_tokenizer)}")
        logger.info(f"New tokenizer size: {len(self.tokenizer)}")

    def setup_special_tokens(self) -> None:
        """设置特殊token"""
        self.special_tokens = {
            "start": self.tokenizer.convert_tokens_to_ids("[CLS]"),
            "sep": self.tokenizer.convert_tokens_to_ids("[SEP]"),
            "pad": self.tokenizer.convert_tokens_to_ids("[PAD]"),
        }

    def process_file(self, file_path: str) -> Dict:
        """处理单个文件的tokenization"""
        with open(file_path, "r", encoding="utf-8") as f:
            sentences = f.read().strip().split("\n")

        # Tokenize sentences
        tokenized_sents = [
            self.tokenizer(s, padding=False, truncation=False)["input_ids"]
            for s in tqdm(sentences, desc="Tokenizing sentences")
        ]

        # Remove CLS tokens
        for sent in tokenized_sents:
            sent.pop(0)

        # Process into fixed-length sequences
        sequences = self._create_sequences(tokenized_sents)
        attention_masks = self._create_attention_masks(sequences)

        return {"input_ids": sequences, "attention_mask": attention_masks}

    def _create_sequences(self, tokenized_sents: List[List[int]]) -> List[List[int]]:
        """创建固定长度的序列"""
        sequences = [[]]
        current_length = 0

        for sent in tokenized_sents:
            sent_length = len(sent)
            position = 0

            while position < sent_length - 1:
                if current_length == 0:
                    sequences[-1].append(self.special_tokens["start"])
                    current_length = 1

                end_pos = (
                    min(
                        sent_length,
                        position + self.config.max_seq_length - current_length,
                    )
                    - 1
                )

                sequences[-1].extend(
                    sent[position:end_pos] + [self.special_tokens["sep"]]
                )
                position = end_pos

                if len(sequences[-1]) == self.config.max_seq_length:
                    sequences.append([])
                current_length = len(sequences[-1])

        return sequences

    def _create_attention_masks(self, sequences: List[List[int]]) -> List[List[int]]:
        """创建attention masks"""
        return [
            [1] * len(seq) + [0] * (self.config.max_seq_length - len(seq))
            for seq in sequences
        ]


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Tokenizer training script")
    parser.add_argument("--train_norm_file", required=True, type=str)
    parser.add_argument("--val_norm_file", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    args = parser.parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_paths = {
        name: save_dir / name
        for name in ["tokenizer_saved", "tokens_saved", "model_saved"]
    }
    for path in save_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    # 初始化trainer
    config = TokenizerConfig()
    trainer = TokenizerTrainer(config)

    # 加载数据集并训练tokenizer
    raw_datasets = trainer.load_datasets(args.train_norm_file, args.val_norm_file)
    training_corpus = trainer.get_training_corpus(raw_datasets)
    trainer.train_tokenizer(training_corpus)

    # 保存tokenizer
    trainer.tokenizer.save_pretrained(save_paths["tokenizer_saved"])
    logger.info("Tokenizer saved successfully")

    # 设置special tokens并处理数据集
    trainer.setup_special_tokens()

    # 处理训练集和验证集
    train_data = trainer.process_file(args.train_norm_file)
    val_data = trainer.process_file(args.val_norm_file)

    # 创建并保存tokenized datasets
    tokenized_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(pd.DataFrame(train_data)),
            "val": Dataset.from_pandas(pd.DataFrame(val_data)),
        }
    )

    tokenized_datasets.save_to_disk(save_paths["tokens_saved"])
    logger.info("Tokenized datasets saved successfully")
    logger.info("Script completed successfully!")


if __name__ == "__main__":
    main()
