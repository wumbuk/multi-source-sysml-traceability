import networkx as nx
import pandas as pd

from kg_retrieve.read_sysml_raw_data import read_sysml_data

# 创建有向图（因为SysML关系通常有方向）
def construct_kg(nodes,edges):
    #初始化知识图谱
    G = nx.DiGraph()

    # 添加节点
    for node in nodes:
        G.add_node(node["ID"], **node)

    # 添加边
    for edge in edges:
        G.add_edge(edge["Source"], edge["Target"], **edge)

    return G
#查看KG信息
def read_kg_information(Graph):
    print("节点列表:")
    for n, data in Graph.nodes(data=True):
        print(n, data)

    print("\n边列表:")
    for u, v, data in Graph.edges(data=True):
        print(u, "->", v, data)

#根据Path获取子系统，将在SysML模型中的属于同一子系统的元素链接到一起，这里的边不表示任何关系

def construct_kg_init(path):
    #查看当前工作的路径
    # print(os.getcwd())

    data_path = path
    df = pd.read_csv(data_path)
    print(df)
    nodes,edges,counters,ss_counter = read_sysml_data(df)
    Graph = construct_kg(nodes,edges)
    # read_kg_information(Graph)

    return Graph,counters,ss_counter


def add_predicted_edge(G,source,target,prediction):
    """
    将预测关系添加到KG中
    :param G:
    :param source:源节点
    :param target:目标节点
    :param prediction:预测信息
    :return:
    """
    G.add_edge(
        source,
        target,
        Type=prediction["Relation"],
        Confidence=prediction["Confidence"],
        Reason=prediction["Reason"],
        Dependency=prediction["Dependency"],
        Source_type=prediction["Source_type"]
    )
