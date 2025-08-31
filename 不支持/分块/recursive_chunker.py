# from 分块.base_chunk import TextSplitter

import re
from typing import Optional, Union, Literal, Any, List
from base_chunk import TextSplitter
def _split_text_with_regex(
    text: str, separator: str, *, keep_separator: Union[bool, Literal["start", "end"]]
) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                ([*splits, _splits[-1]])
                if keep_separator == "end"
                else ([_splits[0], *splits])
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class RecursiveTokenChunker(TextSplitter):
    """Splitting text by recursively look at characters."""
    def __init__(
        self,
        separators: Optional[list[str]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = True,  # noqa: FBT001,FBT002
        is_separator_regex: bool = False,  # noqa: FBT001,FBT002
        **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", "。", ""]
        self._is_separator_regex = is_separator_regex



    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, _separator, keep_separator=self._keep_separator
        )

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks


    def split_text(self, text: str) -> list[str]:
        """Split the input text into smaller chunks based on predefined separators.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)


    def _split_text_with_index(self, text: str, separators: list[str]) -> list[dict]:
        """Split incoming text and return chunks with start/end indices."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(
            text, _separator, keep_separator=self._keep_separator
        )

        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_texts = self._new_merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_texts)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(
                        {"text": s, "start_idx": text.find(s), "end_idx": text.find(s) + len(s)}
                    )
                else:
                    other_info = self._split_text_with_index(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_texts = self._new_merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_texts)
        return final_chunks

    def split_text_with_index(self, text: str) -> list[dict]:
        """Public API: split text into chunks with start/end indices."""
        return self._split_text_with_index(text, self._separators)



if __name__ == "__main__":
    text = "人工智能正在快速发展"

    chunker = RecursiveTokenChunker(
        chunk_size=10,
        chunk_overlap=2,  # 改小
        separators=["\n\n", "\n", " "]
    )

    chunks = chunker.split_text_with_index(text)
    for c in chunks:
        print(c)

    # 加载 BGE-M3 tokenizer
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    #
    # text = "人工智能正在快速发展。尤其是在自然语言处理领域。大语言模型正在改变人机交互的方式。"
    #
    # # 初始化分词器
    # chunker = RecursiveTokenChunker(
    #     chunk_size=15,  # 每个块最大长度
    #     chunk_overlap=5,  # 块之间重叠长度
    #     keep_separator="end"  # 分隔符拼到前一段
    # )
    #
    # # --- 普通切分 ---
    # chunks = chunker.split_text(text)
    # print("=== split_text ===")
    # for c in chunks:
    #     print(c)
    #
    # # --- 带索引的切分 ---
    # chunks_with_index = chunker.split_text_with_index(text)
    # print("\n=== split_text_with_index ===")
    # for c in chunks_with_index:
    #     print(c)
    #     # 验证切出来的片段和原文对应
    #     print("原文对齐:", text[c["start_idx"]:c["end_idx"]])
    #     print()

    # # 初始化一个 Chunker
    # chunker = RecursiveTokenChunker(
    #     chunk_size=20,  # 每个块最多 20 字符
    #     chunk_overlap=5,  # 相邻块有 5 字符重叠
    #     separators=[
    #         "\n\n",  # 段落
    #         "\n",  # 换行
    #         "。", "？", "！", "；", "．",  # 中文句子
    #         ".", "?", "!", ";",  # 英文句子
    #         " ",  # 空格
    #         ""  # 保底：按字符
    #     ],
    #     keep_separator=True
    # )
    #
    # # 待切文本
    # text = """这是第一段，很长很长很长，需要切分。
    # 这是第二段。它也可能很长，所以要继续切分。
    # This is an English sentence. It should also be split correctly!
    # """
    #
    # # 调用切分
    # chunks = chunker.split_text(text)
    #
    # # 打印结果
    # for i, c in enumerate(chunks, 1):
    #     print(f"Chunk {i}: {c}")
    #
    # def token_length_fn(text: str) -> int:
    #     """用 BGE-M3 tokenizer 计算 token 数"""
    #     return len(tokenizer.encode(text, add_special_tokens=False))
    # # print(tokenizer.tokenize("发展，尤其是在自然语"))
    #
    # text = "   i am god . 人工智能  正在快速发展，尤其是在自然语言处理领域。大语言模型正在改变人机交互的方式。"
    # # print(len(text))
    #
    # splitter = RecursiveTokenChunker(
    #     separators=["。", "，", " ", ""],
    #     chunk_size=20,           # 按 token 数切
    #     chunk_overlap=5,         # 按 token 数 overlap
    #     length_function=token_length_fn
    # )
    #
    # print("=== 无 overlap ===")
    # for c in splitter.split_text_with_indices(text, return_overlap=False):
    #     print(c, "token_len:", token_length_fn(c["text"]))
    #
    # print("\n=== 有 overlap ===")
    # for c in splitter.split_text_with_indices(text, return_overlap=True):
    #     print(c, "token_len:", token_length_fn(c["text"]))
    #
    #
