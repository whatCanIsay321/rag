
from typing import Annotated
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

# ===== 定义 State =====
class State(BaseModel):
    step: Annotated[str, "当前步骤"]


# ===== 定义节点基类 =====
class BaseNode:
    def __init__(self, name: str, next_node: str | None = None):
        self.name = name
        self.next_node = next_node

    def __call__(self, state: State) -> Command[State]:
        print(f"进入 {self.name}")
        return Command(
            update={"step": f"{self.name} 完成"},
            goto=self.next_node or END
        )


# ===== 构建 LangGraph =====
def build_graph():
    builder = StateGraph(State)

    # 包装成类
    node_a = BaseNode("node_a", "node_b")
    node_b = BaseNode("node_b", "node_c")
    node_c = BaseNode("node_c", "node_d")
    node_d = BaseNode("node_d", None)  # 最后一个，跳到 END

    builder.add_node("node_a", node_a)
    builder.add_node("node_b", node_b)
    builder.add_node("node_c", node_c)
    builder.add_node("node_d", node_d)

    builder.add_edge(START, "node_a")

    return builder.compile()


# ===== 测试运行 =====
if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke({"step": "开始"})
    print("最终结果：", result)
