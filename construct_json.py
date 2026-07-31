import json


def construct_inf_json(Graph,model_node,req_node,top_edges=None,system_nodes=None,relations_set=None):
    # 构建判断案例列表
    judge_cases = []
    for edge, score in top_edges:
        # 获取源节点和目标节点的详细信息（直接从 Graph 中获取，保证完整）
        source_node = Graph.nodes[edge["Source"]]
        target_node = Graph.nodes[edge["Target"]]
        case_info = {
            "源节点ID": edge["Source"],
            "源节点名称": source_node.get("Name", ""),
            "源节点建模路径":source_node.get("Path", ""),
            "源节点功能描述":source_node.get("Description", ""),
            "源节点类型": source_node.get("NType", ""),
            "源节点属性": source_node.get("Property", ""),
            "目标节点ID": edge["Target"],
            "目标节点名称": target_node.get("Name", ""),
            "目标节点建模路径":target_node.get("Path", ""),
            "目标节点功能描述":target_node.get("Description", ""),
            "目标节点类型": target_node.get("NType", ""),
            "目标节点属性": target_node.get("Property", ""),
            "关系类型": edge.get("EType", ""),
            "关系成立原因": edge.get("Reason",""),
            "来源类型": edge.get("Source_type", ""),
            "置信度": edge.get("Confidence", 0),
            "综合得分": score
        }
        judge_cases.append(case_info)

    # 构建同一子系统中的节点信息列表
    subsystem_nodes = []
    for node in system_nodes:
        node_info = {
            "名称": node.get("Name", ""),
            "功能描述": node.get("Description", ""),
            "建模路径": node.get("Path", ""),
            "元素属性": node.get("Property", ""),
            "元素类型": node.get("NType", "")
        }
        subsystem_nodes.append(node_info)

    # 构建最终输出字典
    output_dict = {
        "模型节点信息": {
            "名称": model_node.get("Name", ""),
            "功能描述": model_node.get("Description", ""),
            "建模路径": model_node.get("Path", ""),
            "元素属性": model_node.get("Property", ""),
            "元素类型": model_node.get("NType", "")
        },
        "需求节点信息": {
            "名称": req_node.get("Name", ""),
            "需求描述": req_node.get("Description", ""),
            "建模路径": req_node.get("Path", ""),
            "需求属性": req_node.get("Property", ""),
            "需求类型": req_node.get("NType", "")
        },
        "可能对追溯关系判断有帮助的案例": judge_cases,
        "和模型节点处于同一子系统中的节点信息": subsystem_nodes,
        "模型元素和需求元素之间可能存在的关系集合": relations_set
    }

    # 输出 JSON 字符串（indent=4 使格式更可读）
    return json.dumps(output_dict, ensure_ascii=False, indent=4)