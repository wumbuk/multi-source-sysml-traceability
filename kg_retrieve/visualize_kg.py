from pyvis.network import Network


def visualize_kg(G, output_file="sysml_kg.html"):
    # 创建网络对象，如果在jupyter中跑，那么notebook=True
    net = Network(
        height="750px",
        width="100%",
        directed=True,
        notebook=False
    )

    # 节点颜色按类型区分（非常重要）
    type_color = {
        # ===== Structural Elements =====
        "Block": "#97C2FC",  # 系统/子系统/模块（主结构）
        "Part_property": "#B4D8FA",  # 组成部件（Block 的组成）

        # ===== Behavioral Elements =====
        "Activity": "#C7B5E8",  # 行为 / 活动（偏紫，区别于结构）

        # ===== Ports / Interfaces =====
        "Control_port": "#FFB347",  # 控制端口（偏橙）
        "Information_port": "#FFD1A1",  # 信息端口（浅橙）
        "Physical_port": "#E9967A",  # 物理端口（偏棕红，强调物理连接）

        # ===== Properties / Parameters =====
        "Value_property": "#A1E3A1",  # 数值属性（可量化参数）

        # ===== Requirements =====
        "Physical_Requirement": "#F6A6A6",  # 物理约束（安全/性能）
        "Functional_Requirement": "#FFD966",  # 功能需求（系统行为）
        "Interaction_Requirement": "#FFE599"  # 交互需求（接口/协同）
    }

    # 添加节点
    for node_id, data in G.nodes(data=True):
        node_type = data.get("NType", "Unknown")
        color = type_color.get(node_type, "#CCCCCC")

        label = node_id

        #将更多信息存储在title中，当鼠标悬停时出现
        title = (
            f"ID:{node_id}\n"
            f"Name:{data.get('Name','')}\n"
            f"NType:{node_type}\n"
            f"Description:{data.get('Description')}\n"
            f"Path:{data.get('Path')}\n"
            f"Property:{data.get('Property')}\n"
        )
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            shape="box",
        )

    # 添加边,要区分原始边和
    for src, tgt, data in G.edges(data=True):
        label = data.get("EType", "")
        #source_type为边的类型，分为关系边和SysML原始结构边
        source_type = data.get("Source_type", "SysML")

        #构造edge title
        title = f"Relation:{label}\n"
        title+= f"Source_type:{source_type}\n"
        if "Confidence"in data:
            title += f"Confidence:{data['Confidence']}\n"
        if data.get("Reason"):
            title += f"Reason:{data.get('Reason', '')}\n"

        if data.get("Dependency"):
            deps = " → ".join(data["Dependency"])
            title += f"Dependency:{deps}\n"

        net.add_edge(
            src,
            tgt,
            label=label,
            title=title,
            arrows="to",
            dashes=(source_type=="LLM")
        )

    net.write_html(output_file,open_browser=True)
