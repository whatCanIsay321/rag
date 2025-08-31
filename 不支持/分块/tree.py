import re
import uuid
import json
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI
from 节点.base_node import NodeLoader


# ========== 数据结构定义 ==========

class Node(BaseModel):
    """文档中的最小单元节点，可以是文本块 / 图片 / 表格"""
    mineru_id: Optional[int] = 0
    id: Optional[int] = 0
    type: str                      # text / image / table
    text: Optional[str] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None
    table: Optional[str] = None
    img_path: Optional[str] = None
    page_idx: Optional[int] = None
    token_count: Optional[int] = 0
    orignal_doc: Optional[str] = None


class ImageNode(BaseModel):
    """图片节点"""
    id: str
    mineru_id: Optional[int] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None
    img_path: Optional[str] = None
    page_idx: Optional[int] = None


class TableNode(BaseModel):
    """表格节点"""
    id: str
    mineru_id: Optional[int] = None
    caption: Optional[str] = None
    footnote: Optional[str] = None
    table: Optional[str] = None
    page_idx: Optional[int] = None


class MetaNode(BaseModel):
    """IndexNode 的元信息"""
    images: Optional[List[ImageNode]] = None
    tables: Optional[List[TableNode]] = None
    page_idx: Optional[List[int]] = None


class IndexNode(BaseModel):
    """树状结构的节点，可以是叶子（text）也可以是总结（summary）"""
    id: str
    node_type: int = 0   # 0=叶子, 1=总结
    summary: Optional[str] = None
    text: Optional[str] = None
    children: Optional[List[str]] = None
    parent: Optional[str] = None
    orignal_doc: Optional[str] = None
    meta: Optional[MetaNode] = None


# ========== LLM summarizer ==========

client = OpenAI(api_key="your_api_key", base_url="http://10.60.200.100:11454/v1")

def summarize_with_llm(text: str, model="Qwen/Qwen3-32B-AWQ") -> str:
    """调用大模型生成总结"""
    prompt = f"请帮我总结以下内容，生成简洁的一段总结：\n\n{text}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    resp = re.sub(r"<think>.*</think>", "", response.choices[0].message.content.strip(), flags=re.DOTALL)
    return resp


# ========== 滑动窗口分组 ==========

def sliding_window_merge(nodes: List[IndexNode], chunk_size: int, overlap: int) -> List[List[IndexNode]]:
    """把节点列表进行滑动窗口分组"""
    merged_groups = []
    step = chunk_size - overlap
    for i in range(0, len(nodes), step):
        group = nodes[i:i + chunk_size]
        if group:
            merged_groups.append(group)
    return merged_groups


# ========== 构建一层树 ==========

def build_one_level(nodes: List[IndexNode], chunk_size: int, overlap: int) -> List[IndexNode]:
    """构建树的一层：合并节点并生成总结节点"""
    groups = sliding_window_merge(nodes, chunk_size, overlap)
    new_nodes = []

    for group in groups:
        combined_text = " ".join([n.text or "" for n in group])
        text = summarize_with_llm(combined_text)

        new_id = str(uuid.uuid4())
        new_node = IndexNode(
            id=new_id,
            node_type=1,
            summary=None,
            text=text,
            children=[n.id for n in group],
            parent=None,
            orignal_doc=group[0].orignal_doc
        )

        # 更新子节点 parent
        for n in group:
            n.parent = new_id

        new_nodes.append(new_node)

    return new_nodes


# ========== 递归构建总结树 ==========

def build_tree(nodes: List[IndexNode], chunk_size: int, overlap: int) -> tuple[IndexNode, dict]:
    """
    递归构建总结树，返回 (root, nodes_dict)
    nodes_dict 包含所有节点（叶子+中间层+根节点）
    """
    level_nodes = nodes
    nodes_dict: dict[str, IndexNode] = {n.id: n for n in nodes}  # 先存叶子
    root = None

    while len(level_nodes) > 1:
        new_nodes = build_one_level(level_nodes, chunk_size, overlap)

        # 存储中间层节点
        for n in new_nodes:
            nodes_dict[n.id] = n

        root = new_nodes[0] if len(new_nodes) == 1 else None
        level_nodes = new_nodes

    return (root or level_nodes[0]), nodes_dict


# ========== 工具：递归打印树 ==========
#
# def print_tree(node: IndexNode, nodes_dict: dict, level: int = 0):
#     indent = "  " * level
#     if node.node_type == 1:
#         print(f"{indent}- [SummaryNode] {node.id}: {node.summary}")
#     else:
#         print(f"{indent}- [LeafNode] {node.id}: {node.text}")
#
#     if node.children:
#         for cid in node.children:
#             child = nodes_dict[cid]
#             print_tree(child, nodes_dict, level + 1)


# ========== 保存 & 加载 ==========

def save_tree_json(root: IndexNode, nodes_dict: dict, path: str):
    """保存树为 JSON 文件"""
    all_nodes = {nid: n.model_dump() for nid, n in nodes_dict.items()}
    data = {
        "root_id": root.id,
        "nodes": all_nodes
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"树已保存到 {path}")


def load_tree_json(path: str) -> tuple[IndexNode, dict]:
    """从 JSON 文件加载树，返回 (root_node, nodes_dict)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_dict: dict[str, IndexNode] = {}
    for nid, node_data in data["nodes"].items():
        nodes_dict[nid] = IndexNode(**node_data)

    root_id = data["root_id"]
    root_node = nodes_dict[root_id]
    return root_node, nodes_dict


# ========== 示例运行 ==========

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    loader = NodeLoader(tokenizer, model_cls=IndexNode)
    nodes = loader.load_for_model(r"D:\新建文件夹\rag-master\final.json")

    # ✅ 一步拿到 root 和完整 nodes_dict
    root, nodes_dict = build_tree(nodes, chunk_size=15, overlap=0)

    print("==== 树结构 ====")
    # print_tree(root, nodes_dict)

    save_tree_json(root, nodes_dict, "tree.json")

    # 加载
    loaded_root, loaded_nodes = load_tree_json("tree.json")
    print("\n==== 从 JSON 加载的树结构 ====")
    # print_tree(loaded_root, loaded_nodes)
