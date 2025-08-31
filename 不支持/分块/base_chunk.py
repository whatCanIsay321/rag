from typing import (
    Callable,
    Iterable,
    List,
    Optional,

)
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)
class BaseChunker(ABC):
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass
#
class TextSplitter(BaseChunker, ABC):
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        """Create a new TextSplitter.

        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of given chunks
            keep_separator: Whether to keep the separator in the chunks
            add_start_index: If `True`, includes chunk's start index in metadata
            strip_whitespace: If `True`, strips whitespace from the start and end of
                              every document
        """
        if chunk_size <= 0:
            msg = f"chunk_size must be > 0, got {chunk_size}"
            raise ValueError(msg)
        if chunk_overlap < 0:
            msg = f"chunk_overlap must be >= 0, got {chunk_overlap}"
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            msg = (
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
            raise ValueError(msg)
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def _new_merge_splits(self, splits: Iterable[str], separator: str) -> List[dict]:
        sep_token_len = self._length_function(separator)  # 用 tokenizer 算 separator 长度

        docs = []
        current_doc: List[str] = []

        token_total = 0  # 当前 chunk 的 token 总数
        start_idx = 0  # 当前 chunk 在全文的字符起始位置


        for d in splits:
            token_len = self._length_function(d)  # token 粒度
            # 判断是否需要切分
            if token_total + token_len + (sep_token_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if token_total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {token_total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append({"text": doc, "start_idx": start_idx, "end_idx": start_idx + len(doc)})

                    # 回退 overlap（用 token 粒度控制）
                    while token_total > self._chunk_overlap or (
                            token_total + token_len + (sep_token_len if len(current_doc) > 0 else 0) > self._chunk_size
                            and token_total > 0
                    ):
                        removed_tokens = self._length_function(current_doc[0]) + (
                            sep_token_len if len(current_doc) > 1 else 0)
                        removed_chars = len(current_doc[0])

                        token_total -= removed_tokens
                        start_idx += removed_chars
                        current_doc = current_doc[1:]

            # 添加当前片段
            current_doc.append(d)
            token_total += token_len + (sep_token_len if len(current_doc) > 1 else 0)

        # 收尾
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append({"text": doc, "start_idx": start_idx, "end_idx": start_idx+len(doc) })

        return docs


class SimpleSplitter(TextSplitter):
    def split_text(self, text: str):
        # 这里用单字符切分，方便演示 overlap
        splits = list(text)
        return self._new_merge_splits(splits, "")


if __name__ == "__main__":
    text = "人工智能正在快速发展。尤其是在自然语言处理领域。大语言模型正在改变人机交互的方式。"
    print(text[0:15])
    splitter = SimpleSplitter(chunk_size=15, chunk_overlap=5)
    chunks = splitter.split_text(text)

    print("=== 分块结果（包含 overlap） ===")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i}: {c}")
