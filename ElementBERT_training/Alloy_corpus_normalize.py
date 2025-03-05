import logging
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from tokenizers.normalizers import BertNormalizer
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextNormalizer:
    """文本标准化处理器类"""

    def __init__(
        self,
        lowercase: bool = False,
        strip_accents: bool = True,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
    ) -> None:
        """
        初始化文本标准化处理器

        Args:
            lowercase: 是否转换为小写
            strip_accents: 是否删除重音符号
            clean_text: 是否清理文本
            handle_chinese_chars: 是否处理中文字符
        """
        self.normalizer = BertNormalizer(
            lowercase=lowercase,
            strip_accents=strip_accents,
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
        )

    def load_corpus(self, file_path: Path) -> List[str]:
        """
        从文件加载语料库

        Args:
            file_path: 语料库文件路径

        Returns:
            包含非空文本行的列表

        Raises:
            FileNotFoundError: 当文件不存在时
            UnicodeDecodeError: 当文件编码错误时
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            logger.error(f"找不到文件: {file_path}")
            raise
        except UnicodeDecodeError:
            logger.error(f"文件编码错误: {file_path}")
            raise

    def normalize_and_export(self, input_path: Path, output_path: Path) -> None:
        """
        标准化文本并导出到文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
        """
        logger.info(f"开始处理文件: {input_path}")

        try:
            texts = self.load_corpus(input_path)
            logger.info(f"已加载 {len(texts)} 行文本")

            normalized_texts = [
                self.normalizer.normalize_str(text)
                for text in tqdm(texts, desc="标准化处理中")
            ]

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(normalized_texts))

            logger.info(f"已保存标准化文本到: {output_path}")

        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            raise


def parse_arguments() -> ArgumentParser:
    """解析命令行参数"""
    parser = ArgumentParser(description="文本标准化处理工具")
    parser.add_argument(
        "--train_corpus_file", required=True, type=Path, help="训练语料库文件路径"
    )
    parser.add_argument(
        "--val_corpus_file", required=True, type=Path, help="验证语料库文件路径"
    )
    parser.add_argument(
        "--train_norm_file", required=True, type=Path, help="标准化后的训练文件保存路径"
    )
    parser.add_argument(
        "--val_norm_file", required=True, type=Path, help="标准化后的验证文件保存路径"
    )
    return parser


def main() -> None:
    """主函数"""
    random.seed(123) #666

    parser = parse_arguments()
    args = parser.parse_args()

    normalizer = TextNormalizer()

    try:
        normalizer.normalize_and_export(args.train_corpus_file, args.train_norm_file)
        normalizer.normalize_and_export(args.val_corpus_file, args.val_norm_file)
        logger.info("所有文件处理完成")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
