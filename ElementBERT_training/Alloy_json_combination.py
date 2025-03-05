import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集配置类"""

    seed: int = 666
    eval_ratio: float = 0.10
    all_data_path: Path = Path("./TiBert_data/All_abstract_data")
    ti_data_path: Path = Path("./TiBert_data/Ti_abstract_data")
    train_output: Path = Path("./train_corpus.txt")
    eval_output: Path = Path("./eval_corpus.txt")


class DatasetSplitter:
    """数据集分割处理类"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        random.seed(config.seed)

    def read_file(self, file_path: Path) -> List[str]:
        """读取文件并处理行"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines()]
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def split_dataset(self, data: List[str]) -> Tuple[List[str], List[str]]:
        """将数据集分割为训练集和验证集"""
        num_eval = round(len(data) * self.config.eval_ratio)
        eval_idx = set(random.sample(range(len(data)), num_eval))

        train_set = []
        eval_set = []

        for idx in tqdm(range(len(data)), desc="Splitting dataset"):
            if idx in eval_idx:
                eval_set.append(data[idx])
            else:
                train_set.append(data[idx])

        return train_set, eval_set

    def save_datasets(self, train_data: List[str], eval_data: List[str]) -> None:
        """保存训练集和验证集到文件"""
        try:
            self._save_lines(self.config.train_output, train_data)
            self._save_lines(self.config.eval_output, eval_data)
        except Exception as e:
            logger.error(f"Error saving datasets: {e}")
            raise

    @staticmethod
    def _save_lines(file_path: Path, lines: List[str]) -> None:
        """将行数据保存到文件"""
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(f"{line}\n")

    def process(self) -> None:
        """执行完整的数据处理流程"""
        try:
            # 读取数据
            logger.info("Reading input files...")
            all_data = self.read_file(self.config.all_data_path)

            # 分割数据集
            logger.info("Splitting dataset...")
            train_set, eval_set = self.split_dataset(all_data)

            # 输出统计信息
            logger.info(
                f"Dataset split complete - "
                f"Total: {len(all_data)}, "
                f"Train: {len(train_set)}, "
                f"Eval: {len(eval_set)}"
            )

            # 保存结果
            logger.info("Saving datasets...")
            self.save_datasets(train_set, eval_set)

            logger.info("Processing completed successfully!")

        except Exception as e:
            logger.error(f"An error occurred during processing: {e}")
            raise


def main():
    """主函数"""
    try:
        # 初始化配置
        config = DatasetConfig()

        # 创建并运行数据集分割器
        splitter = DatasetSplitter(config)
        splitter.process()

    except Exception as e:
        logger.error(f"Script failed: {e}")
        raise


if __name__ == "__main__":
    main()
