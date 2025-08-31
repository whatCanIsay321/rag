
from typing import List, Dict, Any
from transformers import AutoTokenizer


class TokenizerChunker:
    """基于 Tokenizer 的文本切分器，支持 chunk_size 和 overlap。"""

    def __init__(
        self,
        tokenizer,
        chunk_size: int = 128,
        chunk_overlap: int = 20,
        strip_whitespace: bool = True,
    ):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_whitespace = strip_whitespace

    def split_text_with_indices(self, text: str, return_overlap: bool = False) -> List[Dict[str, Any]]:
        """
        切分文本，返回每个 chunk 的内容和在原文中的字符索引。
        """
        # === Step 1: tokenizer 编码 ===
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,  # 每个 token 在原文中的字符 span
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        # === Step 2: 滑动窗口切分 token ===
        results = []
        start_idx = 0
        while start_idx < len(input_ids):
            end_idx = min(start_idx + self.chunk_size, len(input_ids))
            chunk_ids = input_ids[start_idx:end_idx]
            chunk_offsets = offsets[start_idx:end_idx]

            # 计算原文位置
            chunk_start = chunk_offsets[0][0]
            chunk_end = chunk_offsets[-1][1]
            chunk_text = text[chunk_start:chunk_end]

            if self.strip_whitespace:
                chunk_text = chunk_text.strip()

            result = {"text": chunk_text, "start": chunk_start, "end": chunk_end}

            if return_overlap:
                if start_idx == 0:
                    result.update({
                        "main_text": chunk_text,
                        "main_start": chunk_start,
                        "main_end": chunk_end
                    })
                else:
                    # 主体内容 = 去掉前 overlap 的 token
                    main_start = chunk_offsets[self.chunk_overlap][0] if self.chunk_overlap < len(chunk_offsets) else chunk_end
                    main_text = text[main_start:chunk_end]
                    result.update({
                        "main_text": main_text,
                        "main_start": main_start,
                        "main_end": chunk_end
                    })

            results.append(result)

            # 下一个窗口位置：往后滑动 (chunk_size - overlap)
            start_idx += self.chunk_size - self.chunk_overlap

        return results

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    text = "人工智能正在快速发展，尤其是在自然语言处理领域。大语言模型正在改变人机交互的方式。"

    splitter = TokenizerChunker(
        tokenizer=tokenizer,
        chunk_size=20,  # 每个 chunk 的 token 数
        chunk_overlap=5  # 相邻 chunk 共享 5 个 token
    )

    print("=== 无 overlap ===")
    for c in splitter.split_text_with_indices(text, return_overlap=False):
        print(c, "token_len:", len(tokenizer.encode(c["text"], add_special_tokens=False)))

    print("\n=== 有 overlap ===")
    for c in splitter.split_text_with_indices(text, return_overlap=True):
        print(c, "token_len:", len(tokenizer.encode(c["text"], add_special_tokens=False)))

