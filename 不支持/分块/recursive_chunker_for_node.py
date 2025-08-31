from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from spliter.src.recursive_chunker import RecursiveTokenChunker
from Node.base_node import  NodeLoader
from transformers import AutoTokenizer
#
import uuid
class Node(BaseModel):
    """
    文档中的最小单元节点，可以是文本块 / 图片 / 表格
    """
    mineru_id: Optional[int] = 0   # mineru 原始 id
    id: Optional[int] = 0          # 新生成的递增编号
    type: str                      # text / image / table
    text: Optional[str] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None
    table: Optional[str] = None
    img_path: Optional[str] = None
    page_idx: Optional[int] = None
    token_count: Optional[int] = 0
    orignal_doc: Optional[str] = None
  # 一个 chunk 可能对应多个页

class ImageNode(BaseModel):
    """
    图片节点
    """
    id: str                         # 节点唯一 id
    mineru_id: Optional[int] = None # 原始 mineru id
    caption: Optional[str] = None   # 图片标题/说明
    footnote: Optional[str] = None  # 图片脚注
    img_path: Optional[str] = None  # 图片路径（可选）
    page_idx: Optional[int] = None  # 所在页码


class TableNode(BaseModel):
    """
    表格节点
    """
    id: str
    mineru_id: Optional[int] = None
    caption: Optional[str] = None   # 表格标题
    footnote: Optional[str] = None  # 表格脚注
    table: Optional[str] = None  # 表格内容（存为 markdown / csv / json）
    page_idx: Optional[int] = None

class MetaNode(BaseModel):
    """
    IndexNode 的元信息
    """
    images: Optional[List[ImageNode]] = None
    tables: Optional[List[TableNode]] = None
    page_idx: Optional[List[int]] = None

class IndexNode(BaseModel):
    id: str        #节点唯一 ID
    node_type: int = 0
    summary: Optional[str] = None         # 总结内容（summary 节点有）
    text: Optional[str] = None            # 原始文本（叶子节点有）
    children: Optional[List[str]] = None  # 子节点 ID 列表
    parent: Optional[str] = None          # 父节点 ID
    orignal_doc: Optional[str] = None
    meta: Optional[MetaNode] = None


import uuid
import json
from typing import List

class JsonDocChunker(RecursiveTokenChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(self._separators)

    def get_node_text(self, node: Node) -> str:
        if node.type == "text":
            return node.text or ""
        elif node.type == "image":
            parts = [p for p in [node.caption, node.footnote] if p]
            return "\n".join(parts)
        elif node.type == "table":
            parts = [p for p in [node.caption, node.footnote, node.table] if p]
            return "\n".join(parts)
        return ""

    def split_json_as_whole(
        self, nodes: List[Node], save_path: str = None
    ) -> List[IndexNode]:
        texts = []
        node_ranges = []
        cursor = 0

        # 记录 node 在拼接文本中的范围
        for node in nodes:
            t = self.get_node_text(node)
            if not t.strip():
                continue
            texts.append(t)
            start = cursor
            end = cursor + len(t)
            node_ranges.append((start, end, node))
            cursor = end # 保证下一个 node 起始不重叠
            print(start,end)

        full_text = "".join(texts)
        print(len(full_text))
        chunks =  self.split_text_with_indices(full_text, return_overlap=False)

        results: List[IndexNode] = []

        for i, c in enumerate(chunks):

            start_pos = c.get("start")
            end_pos = c.get("end")
            # end_pos = start_pos + len(c)
            # chunk_cursor = end_pos  # 推进到下一个 chunk 的起始位置
            # print(start_pos,end_pos)
            related_ids, related_images, related_tables, related_pages = [], [], [], []

            for start, end, node in node_ranges:
                # print(start,end,start_pos,end_pos)
                if end <= start_pos:# node 在 chunk 左边
                    # print("continue")
                    continue
                if start >= end_pos:
                    # print("break")# node 在 chunk 右边，后面都不会相交
                    break

                # 有交集
                related_ids.append(str(node.id))
                if node.page_idx is not None:
                    related_pages.append(node.page_idx)

                if node.type == "image":
                    related_images.append(
                        ImageNode(
                            id=str(node.id),
                            mineru_id=node.mineru_id,
                            caption=node.caption,
                            footnote=node.footnote,
                            img_path=node.img_path,
                            page_idx=node.page_idx,
                        )
                    )
                elif node.type == "table":
                    related_tables.append(
                        TableNode(
                            id=str(node.id),
                            mineru_id=node.mineru_id,
                            caption=node.caption,
                            footnote=node.footnote,
                            table=node.table,
                            page_idx=node.page_idx,
                        )
                    )

            # ✅ 构建 MetaNode
            meta = MetaNode(
                images=related_images or None,
                tables=related_tables or None,
                page_idx=list(set(related_pages)) or None
            )

            # ✅ 构建 IndexNode
            new_node = IndexNode(
                id=str(uuid.uuid4()),
                summary=None,
                text=c.get("text"),
                children=related_ids or None,
                parent=None,
                meta=meta
            )
            results.append(new_node)

        # ✅ 如果提供了 save_path，则保存到本地 JSON 文件
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(
                    [node.model_dump() for node in results],
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        return results



if __name__=="__main__":
    separators = [
        "\n\n",  # 段落分隔符（空行，优先按段落切分）
        "\n",  # 换行符（次级切分）
        # ===== 中文常用标点 =====
        "。",  # 中文句号
        "？",  # 中文问号
        "！",  # 中文感叹号
        "；",  # 中文分号
        "．",  # 全角句号（部分中文文本中会出现，比如全角输入法）
        # ===== 英文常用标点 =====
        ".",  # 英文句号
        "?",  # 英文问号
        "!",  # 英文感叹号
        ";",  # 英文分号

        # ===== 特殊字符 =====
        "\u200b",  # 零宽空格（Zero-Width Space，用于泰语、缅甸语、柬埔寨语等没有空格分词的语言）
        " ",  # 英文空格（保证英文单词不会被强行黏在一起）
        ""  # 兜底（保证即使没有找到分隔符也能切分，防止死循环）
    ]

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    # 1. 默认用 Node
    loader = NodeLoader(tokenizer)
    # 模拟 JSON 数据
    json_data = [
        {"type": "text", "text": "这是第一页的正文。", "page_idx": 0},
        {"type": "image", "img_path": "images/xxx.jpg",
         "image_caption": ["这是图片的说明"], "image_footnote": [], "page_idx": 0},
        {"type": "table", "img_path": "images/yyy.jpg",
         "table_caption": ["表格标题", "供电公司：新密营销部"],
         "table_footnote": ["表格脚注"], "table_body": "表格内容", "page_idx": 0}
    ]
    nodes = loader.load_for_model(r"D:\PycharmProjects\rag\llm\new.json")
    # 转换成 Node
    # nodes = []
    # for item in json_data:
    #     if item["type"] == "text":
    #         nodes.append(TextNode(**item))
    #     elif item["type"] == "image":
    #         nodes.append(ImageNode(**item))
    #     elif item["type"] == "table":
    #         nodes.append(TableNode(**item))

    # 初始化切分器
    chunker = JsonDocChunker(chunk_size=512,chunk_overlap=50,separators = separators)  # 设定每个 chunk 最大长度 20

    # 执行整体切分
    results = chunker.split_json_as_whole(nodes,"final.json")
    print(len(results))

    # 输出结果
    # for r in results:
    #     print("Chunk:", r["chunk"])
    #     print("Node indices:", r["node_indices"])
    #     print("Page indices:", r["page_indices"])
    #     print("-----")