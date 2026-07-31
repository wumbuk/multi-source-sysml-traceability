"""原始数据格式为,
    id:用于表示节点在知识图谱中的唯一标识
    type:表示元素类型，分为两大类：需求和模型。需求和模型又分为很多小类
    需求：先暂定为Requirement
    模型：分为Block,Activity,State,
    name,
    description
"""
from collections import defaultdict
import pandas as pd

#读取节点
nodes = [
    {"ID": "B1", "NType": "Block", "Name": "Propulsion", "Description": "推进系统","Path": "[1 Problem Domain:2 White Box:1 System]","Property":""},
    {"ID": "B2", "NType": "Block", "Name": "FuelTank", "Description": "燃料罐","Path": "[1 Problem Domain:2 White Box:1 System]","Property":""},
    {"ID": "P1", "NType": "Parameter", "Name": "FuelCapacity", "Description": "燃料容量参数","Path": "[1 Problem Domain:2 White Box:1 System]","Property":"L(升)"},
    {"ID": "C1", "NType": "Physical Requirement", "Name": "MaxWeight", "Description": "最大重量限制","Path": "[1 Problem Domain:1 Black Box:1 Staker holder Requirement]","Property":""},
    {"ID": "R1", "NType": "Physical Requirement", "Name": "Req1", "Description": "系统必须保证最大重量","Path": "[1 Problem Domain:1 Black Box:1 Staker holder Requirement]","Property":""},
    {"ID": "R2", "NType": "Physical Requirement", "Name": "Req2", "Description": "系统必须保证推进功率","Path": "[1 Problem Domain:1 Black Box:1 Staker holder Requirement]","Property":""}
]

#获取边的关系

edges = [
    {"Source": "B1", "Target": "P1", "EType": "hasParameter","Confidence":"" ,"Reason":"","Source_type":"SysML"},
    {"Source": "B2", "Target": "P1", "EType": "hasParameter","Confidence":"" ,"Reason":"","Source_type":"SysML"},
    {"Source": "R1", "Target": "C1", "EType": "Satisfies","Confidence":"" ,"Reason":"","Source_type":"LLM"},
    {"Source": "B1", "Target": "R2", "EType": "Satisfies","Confidence":"" ,"Reason":"","Source_type":"LLM"},
]

TYPE_PREFIX = {
    "Block": "B",
    "Parameter": "P",
    "State": "S",
    "Action": "A",
    "Control_port":"CP",
    "Information_port":"IP",
    "physical_port":"PP",
    "Part_property":"PAP",
    "Value_property":"VP",
    "Constraint_property":"CPR",
    "Constraint_Requirement": "CR",
    "Physical_Requirement": "PR",
    "Functional_Requirement": "FR",
    "Interaction_Requirement": "IR",
    "Interface_Requirement": "IFR",
}


#对节点进行去重操作
def drop_duplicates(df, subset):
    """
    :param df:需要去重的数据集合
    :param subset:去重参考的属性列
    :return:去重过后的数据集
    """
    return df.drop_duplicates(subset=subset,keep='first')
#获取边的源id，通过数据集中的元素对中的信息进行匹配来获得
def get_source_id(nodes,row):
    """

    :param nodes:节点列表
    :param row: 行信息
    :return: 返回Source_ID
    """
    for data in nodes:
        if (
            data["NType"] == row["typeA"] and
            data["Name"] == row["NameA"] and
            data["Description"] == row["contextA"] and
            data["Path"] == row["pathA"] and
            (
                (pd.isna(data["Property"]) and pd.isna(row["propertyA"])) or
                (data["Property"] == row["propertyA"])
            )
        ):
            return data["ID"]
    return None
#获取边的目标id，通过数据集中的元素对中的信息进行匹配来获得
def get_target_id(nodes, row):
    for data in nodes:
        if (
            data["NType"] == row["typeB"] and
            data["Name"] == row["NameB"] and
            data["Description"] == row["contextB"] and
            data["Path"] == row["pathB"] and
            (
                (pd.isna(data["Property"]) and pd.isna(row["propertyB"])) or
                (data["Property"] == row["propertyB"])
            )
        ):
            return data["ID"]
    return None
#将关系提取出来
def extract_relation_type(relation):
    if isinstance(relation, str) and relation.startswith("1/"):
        return relation.split("/", 1)[1]
    return None

#处理property为nan的情况
def normalize_property(x):
    return None if pd.isna(x) else x

#根据Path提取子系统信息,不依赖固定层级数
import re

def extract_subsystem_from_path(path: str):
    """
    从 Path 字符串中提取子系统名称（最后一级）
    """
    if not isinstance(path, str):
        return None

    # 去掉首尾 []
    path = path.strip("[]")

    # 按 :: 分割
    parts = path.split("::")
    if not parts:
        return None

    # 最后一段形如：1 Cooling 或 Cooling
    last_part = parts[-1].strip()

    # 去掉前面的编号
    match = re.search(r"\s*(.+)$", last_part)
    return match.group(1).strip() if match else last_part


#从节点的路径种取出节点所属子系统
from collections import OrderedDict

def build_subsystem_nodes(nodes):
    """
    根据 nodes 中的 Path 构建去重后的子系统节点
    """
    subsystem_map = OrderedDict()  # 保序 + 去重
    ss_counter = 0

    for node in nodes:
        path = node.get("Path")
        subsystem_name = extract_subsystem_from_path(path)

        if not subsystem_name:
            continue

        if subsystem_name not in subsystem_map:
            ss_counter += 1
            subsystem_id = f"SS{ss_counter}"
            subsystem_map[subsystem_name] = {
                "ID": subsystem_id,
                "NType": "Subsystem",
                "Name": subsystem_name,
                "Description": "",
                "Path": "",
                "Property": "",
            }

    return list(subsystem_map.values()),ss_counter

def build_subsystem_index(subsystem_nodes):
    """
    构建子系统名称到子系统ID的映射
    """
    return {ss["Name"]: ss["ID"] for ss in subsystem_nodes}

def build_subsystem_edges(nodes, subsystem_nodes):
    """
    为每个节点，根据 Path 生成 belongsToSubsystem 边
    """
    subsystem_index = build_subsystem_index(subsystem_nodes)
    edges = []

    for node in nodes:
        path = node.get("Path")
        subsystem_name = extract_subsystem_from_path(path)

        if not subsystem_name:
            continue

        subsystem_id = subsystem_index.get(subsystem_name)
        if not subsystem_id:
            continue

        edge = {
            "Source": node["ID"],
            "Target": subsystem_id,
            "EType": "belongsToSubsystem",
            "Reason": "Path_Derived",
            "Confidence": "1",
            "Source_type": "SysML"
        }
        edges.append(edge)

    return edges

# 工具函数区（和 drop_duplicates 同一层级）
def deduplicate_edges(edges):
    seen = set()
    unique_edges = []

    for e in edges:
        key = (e["Source"], e["Target"], e["EType"])
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    return unique_edges

def add_node(Graph=None, counters=None, df=None):
    nodes = []

    nodes_element_A = df.iloc[:, [0, 1, 2, 3, 4]]
    nodes_element_B = df.iloc[:, [5, 6, 7, 8, 9]]

    nodes_element_A = drop_duplicates(
        df=nodes_element_A,
        subset=["NameA", "pathA", "contextA", "propertyA", "typeA"]
    )
    # print("元素A的数量为：", len(nodes_element_A))

    nodes_element_B = drop_duplicates(
        df=nodes_element_B,
        subset=["NameB", "pathB", "contextB", "propertyB", "typeB"]
    )
    # print("元素B的数量为：", len(nodes_element_B))

    # =========================
    # ✅ 1️⃣ 构建已有节点 key（支持初始化模式）
    # =========================
    existing_node_keys = set()

    if Graph is not None and len(Graph.nodes) > 0:
        for _, data in Graph.nodes(data=True):
            key = (
                data.get("Name"),
                data.get("Description"),
                data.get("Path"),
                data.get("Property"),
                data.get("NType"),
            )
            existing_node_keys.add(key)

    # =========================
    # ✅ 2️⃣ key 构造函数
    # =========================
    def build_key(name, desc, path, prop, ntype):
        return (name, desc, path, prop, ntype)

    # =========================
    # ✅ 3️⃣ 统一处理函数（消除重复代码）
    # =========================
    def process_rows(rows, col_map):
        for _, row in rows.iterrows():
            node_type = row[col_map["type"]]
            prefix = TYPE_PREFIX.get(node_type, node_type[0].upper())

            key = build_key(
                row[col_map["name"]],
                row[col_map["desc"]],
                row[col_map["path"]],
                row[col_map["prop"]],
                node_type
            )

            # ❗统一去重逻辑（初始化 / 增量 都适用）
            if key in existing_node_keys:
                continue

            counters[prefix] += 1
            node_id = f"{prefix}Node{counters[prefix]}"

            node = {
                "ID": node_id,
                "NType": node_type,
                "Name": row[col_map["name"]],
                "Description": row[col_map["desc"]],
                "Path": row[col_map["path"]],
                "Property": row[col_map["prop"]],
            }

            nodes.append(node)
            existing_node_keys.add(key)  # 防止本批次重复

    # =========================
    # ✅ 4️⃣ 执行
    # =========================
    process_rows(nodes_element_A, {
        "name": "NameA",
        "desc": "contextA",
        "path": "pathA",
        "prop": "propertyA",
        "type": "typeA"
    })

    process_rows(nodes_element_B, {
        "name": "NameB",
        "desc": "contextB",
        "path": "pathB",
        "prop": "propertyB",
        "type": "typeB"
    })

    return nodes

#读取数据形成形成节点和边集合，这里的数据集，是已经经过处理的格式
def read_sysml_data(df):
    """
    :param df:参数df存储的是panda读取的文件
    :return:nodes,edges
    nodes包含的信息ID(自动生成)，Type对应df中的type，Name对应df中的Name，Description对应df中的context，Path对应df中的Path，Property对应df中的property
    edges包含的信息为Source，Target来源于nodes中两端节点的ID，Type从df中的Relation获取，Reason是根据Source_type来确定的，如果是LLM生成，就将LLM的判断判断原因写入，否则就标为SysML_Instance,Confidence为置信度分数
    """
    counters = defaultdict(int)  # 每种 Type 单独编号
    edges=[]
    nodes=add_node(counters=counters,df=df)
    # print("nodes:",len(nodes))
    #构造边节点集合
    # edges包含的信息为Source，Target来源于nodes中两端节点的ID，Type从df中的Relation获取，Reason是根据Source_type来确定的，如果是LLM生成，就将LLM的判断判断原因写入，否则就标为SysML_Instance,
    # 初始构建的KG的信息来源均是SysML
    #寻找边的策略是根据数据集中的
    for _,row in df.iterrows():
        relation = row["Relation"]
        if isinstance(relation, str) and relation.startswith("1/"):
            relation_type = relation.split("/", 1)[1]
            source_id = get_source_id(nodes=nodes, row=row)
            target_id = get_target_id(nodes=nodes, row=row)
            edge = {
                "Source": source_id,
                "Target": target_id,
                "EType": relation_type,
                "Reason":"SysML_Instance",
                "Confidence": "1",
                "Source_type":"SysML"
            }
            edges.append(edge)

    # 已有：nodes（模型 + 需求节点）
    # 第一步：构建子系统节点
    subsystem_nodes,ss_counter = build_subsystem_nodes(nodes)
    # print("subsystem_nodes:",subsystem_nodes)
    # 第二步：构建 子系统归属关系边
    subsystem_edges = build_subsystem_edges(nodes, subsystem_nodes)
    # 合并进原有 KG
    all_nodes = nodes + subsystem_nodes
    all_edges = edges + subsystem_edges
    all_edges = deduplicate_edges(all_edges)
    # print("初始化总节点数量为:", len(all_nodes))

    return all_nodes,all_edges,counters,ss_counter#这里也返回计数器，便于后续节点的添加


# if __name__ == "__main__":
#     data_path = r"../Data/Problem_domain_Black Box&White Box.csv"
#     df = pd.read_csv(data_path)
#     nodes,edges = read_sysml_data(df)

def build_node_pairs(df, Graph=None, counters=None):
    node_pairs = []
    pair_index = {}

    # =========================
    # ✅ 1️⃣ 构建 Graph 节点索引（语义去重核心）
    # =========================
    existing_node_map = {}

    if Graph is not None:
        for nid, data in Graph.nodes(data=True):
            key = (
                normalize_value(data.get("Name")),
                normalize_value(data.get("Description")),
                normalize_value(data.get("Path")),
                normalize_value(data.get("Property")),
                normalize_value(data.get("NType")),
            )
            existing_node_map[key] = nid

    # =========================
    # ✅ 2️⃣ 统一构造 key
    # =========================
    def build_key(node):
        return (
            normalize_value(node.get("Name")),
            normalize_value(node.get("Description")),
            normalize_value(node.get("Path")),
            normalize_value(node.get("Property")),
            normalize_value(node.get("NType")),
        )

    # =========================
    # ✅ 3️⃣ 获取或创建ID
    # =========================
    def get_or_create_id(node):
        key = build_key(node)

        # ✅ 已存在 → 复用ID
        if key in existing_node_map:
            return existing_node_map[key]

        # ❗不存在 → 创建新ID
        node_type = node["NType"]
        prefix = TYPE_PREFIX.get(node_type, node_type[0].upper())

        counters[prefix] += 1
        new_id = f"{prefix}Node{counters[prefix]}"

        # ⚠️ 加入 map（防止本批重复）
        existing_node_map[key] = new_id

        return new_id

    # =========================
    # ✅ 4️⃣ 主循环
    # =========================
    for _, row in df.iterrows():

        # ===== A节点 =====
        nodeA = {
            "NType": row["typeA"],
            "Name": row["NameA"],
            "Description": row["contextA"],
            "Path": row["pathA"],
            "Property": row["propertyA"],
        }

        nodeA_id = get_or_create_id(nodeA)
        nodeA["ID"] = nodeA_id

        # ===== B节点 =====
        nodeB = {
            "NType": row["typeB"],
            "Name": row["NameB"],
            "Description": row["contextB"],
            "Path": row["pathB"],
            "Property": row["propertyB"],
        }

        nodeB_id = get_or_create_id(nodeB)
        nodeB["ID"] = nodeB_id

        pair = {"A": nodeA, "B": nodeB}
        node_pairs.append(pair)

        pair_key = (nodeA_id, nodeB_id)
        pair_index[pair_key] = pair

    return node_pairs, pair_index


import math

def normalize_value(v):
    """强规范化（用于KG去重）"""

    # ===== 1️⃣ 空值处理 =====
    if v is None:
        return ""

    if isinstance(v, float) and math.isnan(v):
        return ""

    v = str(v)

    # 处理字符串 "nan"
    if v.lower() == "nan":
        return ""

    # ===== 2️⃣ 去首尾空格 =====
    v = v.strip()

    # ===== 3️⃣ 多空格压缩（核心！！）=====
    v = re.sub(r"\s+", " ", v)

    # ===== 4️⃣ 去路径噪声（可选但强烈建议）=====
    # 去掉结尾多余符号
    v = v.strip("[]")

    # ===== 5️⃣ 小写统一 =====
    v = v.lower()

    return v


def build_node_key(node):
    """
    构建鲁棒去重 key（核心）
    """
    return (
        normalize_value(node.get("Name")),
        normalize_value(node.get("Path")),
        normalize_value(node.get("NType")),
        normalize_value(node.get("Property")),
    )
    # ❗不使用 Description（太容易轻微变化）


def add_nodes_from_pairs(Graph, node_pairs):
    """
    从 node_pairs 加入节点（强力去重版）
    """
    if Graph is None:
        return

    # =========================
    # ✅ 1️⃣ 构建已有节点 key
    # =========================
    existing_keys = set()

    for _, data in Graph.nodes(data=True):
        key = build_node_key(data)
        existing_keys.add(key)

    added = 0
    skipped = 0

    # =========================
    # ✅ 2️⃣ 遍历 node_pairs
    # =========================
    for pair in node_pairs:
        for side in ["A", "B"]:

            node = pair.get(side)
            if not node:
                continue

            key = build_node_key(node)

            if key in existing_keys:
                skipped += 1
                continue

            # 加入图
            Graph.add_node(node["ID"], **node)
            existing_keys.add(key)
            added += 1

    print(f"[节点加入完成] 新增: {added} | 去重跳过: {skipped}")



#更新子系统节点和子系统边到Graph中
def update_subsystem_nodes(Graph, nodes, ss_counter):
    """
    增量构建子系统节点（去重 + 复用已有）
    """

    # =========================
    # ✅ 1️⃣ 已有子系统索引
    # =========================
    subsystem_index = {}  # name -> ID

    for node_id, data in Graph.nodes(data=True):
        if data.get("NType") == "Subsystem":
            name_key = normalize_value(data.get("Name"))
            subsystem_index[name_key] = node_id

    added = 0

    # =========================
    # ✅ 2️⃣ 遍历新节点
    # =========================
    for node in nodes:
        path = node.get("Path")
        subsystem_name = extract_subsystem_from_path(path)

        if not subsystem_name:
            continue

        key = normalize_value(subsystem_name)

        # 已存在 → 跳过
        if key in subsystem_index:
            continue

        # 新建子系统节点
        ss_counter += 1
        ss_id = f"SS{ss_counter}"

        ss_node = {
            "ID": ss_id,
            "NType": "Subsystem",
            "Name": subsystem_name.strip(),
            "Description": "",
            "Path": "",
            "Property": "",
        }

        Graph.add_node(ss_id, **ss_node)
        subsystem_index[key] = ss_id
        added += 1

    print(f"[子系统节点] 新增: {added}")

    return subsystem_index, ss_counter

def update_subsystem_edges(Graph, nodes, subsystem_index):
    """
    构建 节点 → 子系统 边（严格去重 + 防隐式建点）
    """

    # =========================
    # ✅ 1️⃣ Graph中真实节点
    # =========================
    valid_nodes = set(Graph.nodes)

    # =========================
    # ✅ 2️⃣ 已有边集合（去重）
    # =========================
    existing_edges = set(
        (u, v)
        for u, v, data in Graph.edges(data=True)
        if data.get("EType") == "belongsToSubsystem"
    )

    added = 0
    skipped_missing_node = 0
    skipped_missing_ss = 0
    skipped_duplicate = 0

    # =========================
    # ✅ 3️⃣ 构建边
    # =========================
    for node in nodes:
        node_id = node.get("ID")

        # ❗关键1：节点必须已存在
        if node_id not in valid_nodes:
            skipped_missing_node += 1
            continue

        path = node.get("Path")
        subsystem_name = extract_subsystem_from_path(path)

        if not subsystem_name:
            continue

        key = normalize_value(subsystem_name)
        ss_id = subsystem_index.get(key)

        # ❗关键2：子系统必须存在
        if not ss_id or ss_id not in valid_nodes:
            skipped_missing_ss += 1
            continue

        edge_key = (node_id, ss_id)

        # ❗关键3：边去重
        if edge_key in existing_edges:
            skipped_duplicate += 1
            continue

        # =========================
        # ✅ 4️⃣ 加边（安全）
        # =========================
        Graph.add_edge(
            node_id,
            ss_id,
            EType="belongsToSubsystem",
            Reason="Path_Derived",
            Confidence=1.0,
            Source_type="SysML"
        )

        existing_edges.add(edge_key)
        added += 1

    # =========================
    # ✅ 5️⃣ 日志（强建议保留）
    # =========================
    print(
        f"[子系统边] 新增: {added} | "
        f"跳过(节点不存在): {skipped_missing_node} | "
        f"跳过(子系统不存在): {skipped_missing_ss} | "
        f"跳过(重复): {skipped_duplicate}"
    )
