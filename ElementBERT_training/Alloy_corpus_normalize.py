import json
import random
import re
from argparse import ArgumentParser

from tokenizers.normalizers import BertNormalizer
from tqdm import tqdm

random.seed(123)

parser = ArgumentParser()
parser.add_argument("--train_corpus_file", required=True, type=str)
parser.add_argument("--val_corpus_file", required=True, type=str)
parser.add_argument("--train_norm_file", required=True, type=str)
parser.add_argument("--val_norm_file", required=True, type=str)
args = parser.parse_args()


def load_corpus(path):
    """
    Reads a text file where each line is a separate paragraph and returns a list of paragraphs.

    Args:
        path (str): The file path to the text file.

    Returns:
        list: A list of paragraphs as strings.
    """
    texts = []  # Initialize an empty list to store the paragraphs.
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            # Strip the line to remove any leading/trailing whitespace and newline characters.
            stripped_line = line.strip()
            # Only add the line if it is not empty.
            if stripped_line:
                texts.append(stripped_line)

    return texts


normalize = BertNormalizer(
    lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True
)


def normalize_export_file(corpus_path, save_path):
    """
    Normalize text and export as TXT file.
    """
    texts = load_corpus(corpus_path)
    corpus_norm = [normalize.normalize_str(sent) for sent in tqdm(texts)]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_norm))


normalize_export_file(args.train_corpus_file, args.train_norm_file)
normalize_export_file(args.val_corpus_file, args.val_norm_file)

print("Script run COMPLETE!!!")

