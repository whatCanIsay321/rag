import json
from typing import List, Optional, Dict, Any, Union, Type
from pathlib import Path

from pydantic import BaseModel, TypeAdapter
from transformers import AutoTokenizer
from 分块.recursive_chunker import RecursiveTokenChunker


# ========== 定义 Node ==========
class MineruNode(BaseModel):
    """
    文档中的最小单元节点，可以是文本块 / 图片 / 表格
    """
    mineru_id: Optional[int] = 0          # mineru 原始 id
    id: Optional[int] = 0                 # 新生成的递增编号
    type: str                             # text / image / table
    text: Optional[str] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None
    table: Optional[str] = None
    img_path: Optional[str] = None
    page_idx: Optional[int] = None
    token_count: Optional[int] = 0
    orignal_doc: Optional[str] = None


# ========== NodeLoader ==========
class NodeLoader:
    """
    Node 加载器：
    - 将 JSON 转换为模型对象（默认 Node，可自定义）
    - 对 text 节点自动分块
    - 对 image/table 节点计算 token
    - 可选合并 mineru_id 相同的 text 节点
    """

    def __init__(
        self,
        tokenizer,
        model_cls: Type[BaseModel] = MineruNode,
        separators=None,
        chunk_size: int = 512,
        over_lap: int = 0
    ):
        self.tokenizer = tokenizer
        self.model_cls = model_cls
        self.chunk_size = chunk_size
        self.over_lap = over_lap
        self.chunker = RecursiveTokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=over_lap,
            separators=separators,
            length_function=self._huggingface_tokenizer_length
        )

    # ====== 工具方法 ======
    def _huggingface_tokenizer_length(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.tokenizer.encode(text)) if text else 0

    def normalize(self, value, sep: str = "\n") -> Optional[str]:
        """将 list / str 统一转成字符串"""
        if value is None:
            return None
        if isinstance(value, list):
            return sep.join(map(str, value)) if value else None
        return str(value)

    # ====== 从 mineru 输出加载 ======
    def load_from_mineru(self, file_path: str) -> List[BaseModel]:
        """
        将 JSON 的 dict list 转换为模型类对象列表
        - text: 超过阈值会自动切分成多个 chunk
        - image/table: 保留 caption/footnote，并计算 token 数
        """
        data = self.load_from_file(file_path)
        nodes, id_count = [], 0

        for item in data:
            node_type = item.get("type")
            mineru_id = item.get("id")

            # --- 文本节点 ---
            if node_type == "text":
                text = item.get("text")
                token_count = self._huggingface_tokenizer_length(text)

                if token_count > self.chunk_size:
                    chunks = self.chunker.split_text(text)
                    for chunk in chunks:
                        nodes.append(self.model_cls(
                            mineru_id=mineru_id,
                            id=id_count,
                            type="text",
                            text=chunk,
                            page_idx=item.get("page_idx"),
                            token_count=self._huggingface_tokenizer_length(chunk),
                        ))
                        id_count += 1
                else:
                    nodes.append(self.model_cls(
                        mineru_id=mineru_id,
                        id=id_count,
                        type="text",
                        text=text,
                        page_idx=item.get("page_idx"),
                        token_count=token_count,
                    ))
                    id_count += 1

            # --- 图片节点 ---
            elif node_type == "image":
                caption = self.normalize(item.get("image_caption"))
                footnote = self.normalize(item.get("image_footnote"))
                content_for_count = " ".join(filter(None, [caption, footnote]))

                nodes.append(self.model_cls(
                    mineru_id=mineru_id,
                    id=id_count,
                    type="image",
                    caption=caption,
                    footnote=footnote,
                    img_path=item.get("img_path"),
                    page_idx=item.get("page_idx"),
                    token_count=self._huggingface_tokenizer_length(content_for_count),
                ))
                id_count += 1

            # --- 表格节点 ---
            elif node_type == "table":
                caption = self.normalize(item.get("table_caption"))
                footnote = self.normalize(item.get("table_footnote"))
                content_for_count = " ".join(filter(None, [caption, footnote]))

                nodes.append(self.model_cls(
                    mineru_id=mineru_id,
                    id=id_count,
                    type="table",
                    caption=caption,
                    footnote=footnote,
                    img_path=item.get("img_path"),
                    page_idx=item.get("page_idx"),
                    token_count=self._huggingface_tokenizer_length(content_for_count),
                ))
                id_count += 1

        return nodes

    # ====== 合并相同 mineru_id 的 text 节点 ======
    def merge_text_nodes(self, nodes: List[BaseModel]) -> List[BaseModel]:
        merged_nodes: List[BaseModel] = []
        current_group: List[BaseModel] = []

        def flush_group():
            if not current_group:
                return
            merged_text = "\n".join([n.text for n in current_group if n.text])
            token_count = self._huggingface_tokenizer_length(merged_text)
            new_node = current_group[0].copy(update={
                "id": current_group[0].mineru_id,
                "token_count": token_count
            })
            merged_nodes.append(new_node)
            current_group.clear()

        for node in nodes:
            if node.type == "text":
                if not current_group or node.mineru_id == current_group[0].mineru_id:
                    current_group.append(node)
                else:
                    flush_group()
                    current_group.append(node)
            else:
                flush_group()
                node.id = node.mineru_id  # 非 text 节点也保证 id 和 mineru_id 一致
                merged_nodes.append(node)

        flush_group()
        return merged_nodes

    # ====== 从 JSON 文件加载 ======
    def load_for_model(self, path: Union[str, "PathLike"]) -> List[BaseModel]:
        """从 JSON 文件加载并转换为模型类对象列表（不做任何 chunk 处理）"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                raw_list = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 格式错误: {path}") from e

        adapter = TypeAdapter(List[self.model_cls])
        return adapter.validate_python(raw_list)

    # ====== 原始 JSON 数据读取 ======
    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ====== 保存到 JSON 文件 ======
    def save_to_file(self, nodes: List[BaseModel], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([node.model_dump() for node in nodes], f, ensure_ascii=False, indent=2)

    # ====== 计算 token ======
    def compute_token(self, nodes: List[BaseModel], field: str) -> List[BaseModel]:
        for node in nodes:
            if node.token_count == 0:
                value = getattr(node, field)
                node.token_count = self._huggingface_tokenizer_length(value)
        return nodes


if "__nam__"=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # 1. 默认用 Node
    loader = NodeLoader(tokenizer)

    nodes = loader.load_from_mineru("xxx_content_list.json")
    print(f"原始节点: {len(nodes)}")

    nodes_merged = loader.merge_text_nodes(nodes)
    print(f"合并后节点: {len(nodes_merged)}")

    loader.save_to_file(nodes_merged, "nodelist.json")

    # 2. 用自定义模型类
    class MyNode(MineruNode):
        extra_field: Optional[str] = None

    loader2 = NodeLoader(tokenizer, model_cls=MyNode)
    nodes2 = loader2.load_for_model("nodelist.json")
    print(type(nodes2[0]))  # <class '__main__.MyNode'>
