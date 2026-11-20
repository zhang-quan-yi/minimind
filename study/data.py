"""第一课：把文本转换成预训练样本。"""

from pathlib import Path

from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def encode_pretrain(
    text: str, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> tuple[Tensor, Tensor]:
    """返回两个形状为 [max_length]、类型为 torch.long 的张量。"""
    raise NotImplementedError("第一课：请实现 encode_pretrain")


def main() -> None:
    tokenizer_path = Path(__file__).resolve().parents[1] / "model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    for text, max_length in [("你好", 6), ("你好，世界", 6), ("你好，世界", 4)]:
        print(f"\n{text!r}, max_length={max_length}")
        input_ids, labels = encode_pretrain(text, tokenizer, max_length)
        print("input_ids:", input_ids.tolist())
        print("labels:   ", labels.tolist())
        print("预测目标:  ", labels[1:].tolist())


if __name__ == "__main__":
    main()
