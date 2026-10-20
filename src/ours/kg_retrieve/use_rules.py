from kg_retrieve.Retreive_Rules import semantic_score,retrieve_node,retrieve_edge, req_semantic_score, edge_final_score
from time import perf_counter

def select_top_nodes_and_scores(tmp_nodes, mod_sim_score):
    k = len(tmp_nodes) // 2

    # 1️⃣ 按 score 排序节点
    sorted_nodes = sorted(
        tmp_nodes,
        key=lambda node: mod_sim_score.get(node["ID"], 0),
        reverse=True
    )

    # 2️⃣ 取 Top-K 节点
    top_nodes = sorted_nodes[:k]

    # 3️⃣ 构建对应的 score dict
    top_scores = {
        node["ID"]: mod_sim_score[node["ID"]]
        for node in top_nodes
    }

    return top_nodes, top_scores

def retrieve(Graph,waitnode,req_node,model):
    #============================对知识图谱的检索==============================================
    #第一步检索，遍历图获取，节点列表信息tmp_nodes，子系统列表信息system_nodes

    #===================Graph-based Context Extraction==========================================
    # [RQ4 TIMING START] 图遍历：获取 tmp_nodes 和 system_nodes
    graph_context_start = perf_counter()
    tmp_nodes,system_nodes=retrieve_node(waitnode,Graph)
    graph_context_ms = (perf_counter() - graph_context_start) * 1000
    # [RQ4 TIMING END] Graph-based Context Extraction

    #============================Case-based-Retrieval====================================
    # [RQ4 TIMING START] 案例检索：语义编码、候选边检索、相似度打分和 Top-K 排序
    case_retrieval_start = perf_counter()
    #第二步检索，获取模型元素名称相似度分数
    mod_sim_score=semantic_score(model,waitnode,tmp_nodes)

    #============================根据分数对需求节点排序，其中一个小组件，可以通过实验检测结果=====================
    # print("tmp_nodes:",tmp_nodes)
    # print("mod_sim_score:",mod_sim_score)
    # element_with_score=list(zip(tmp_nodes,))
    tmp_nodes,mod_sim_score=select_top_nodes_and_scores(tmp_nodes, mod_sim_score)
    # print("============================after_sort=====================================")
    # print("tmp_nodes:",tmp_nodes)
    # print("mod_sim_score:",mod_sim_score)
    #========================进行最终的分数组合并据此筛选出边信息=============================
    #第三步检索
    # 接下来就是，对tmp_nodes中的节点，已经产生的追溯关系，进行查找，也就是查找边，
    #从第一个检索规则中得来的节点集合进行追溯关系的检索，
    # - SysML 原生结构边（最高可信）
    # - LLM 推理边（可用但可信度低），后续可以人工验证，如果正确，那么就可以标注为SysML
    # 然后根据tmp_edges中的边两端的节点的相似度分数相加和，并根据追溯关系的来源Source_Type进行加权，依次得到每条边的分数集合end_scores
    ## #在第三步的检索中，首先根据已经收集的tmp_nodes节点，检索它们的边，然后找另一端的需求节点和待判需求节点相同类型的边，把这些边整合到一个新的集合tmp_edges
    # 在得到的tmp_edges中的边的Target节点（需求节点）中，找到和req_node具有相同["NType"]值的需求节点集合req_nodes，然后进一步得出由tmp_nodes和req_nodes中的节点连接到的边集合tmp_edges
    tmp_edges,req_nodes=retrieve_edge(tmp_nodes,req_node,Graph)


    # 再计算req_nodes中的节点名称n["Name"]和req_node["Name"]的相似度分数req_sim_score
    req_sim_score=req_semantic_score(model,req_node,req_nodes)

    # 然后根据tmp_edges中的边两端的节点的相似度分数相加和，并根据追溯关系的来源Source_Type进行加权，依次得到每条边的分数集合end_scores
    final_scores=edge_final_score(mod_sim_score,req_sim_score,tmp_edges)

    # print("final_scores:",final_scores)
    #最终有用的信息是tmp_edges,tmp_nodes,req_nodes,text,system_nodes,waitnode,req_node
    #组合方式，其中req_node,wait_node表示待判的节点对
    # tmp_edges中提取出边，tmp_nodes，req_nodes中提取出边所对应的节点，作为关系判断可能的证据信息
    # text作为小模型的初步关系判断及理由
    # system_nodes作为提取出的和waitnode在同一模块中的节点集合，它们说明waitnode所在的是怎么样的一个模块
    # result作为ANode1和FRNode1之间可能存在的关系集合
    # {
    #     模型节点信息:名称：waitnode["Name"],信息描述:waitnode["Description"],建模路径:waitnode["Path"],元素属性:waitnode["Property"],元素类型:waitnode["NType"],
    #     需求节点信息: 名称：req_node["Name"], 信息描述: req_node["Description"], 建模路径: req_node["Path"], 元素属性: req_node["Property"], 元素类型: req_node["NType"],
    #     判断案例：
    #     {依次遍历final_edges，获取所有的top-k边信息，在这里展示}
    #     初步判断：
    #     {text}
    #     和模型节点处于同一子系统中的节点信息：
    #     {遍历system_nodes得到所有的节点信息
    #         节点1信息:名称：waitnode["Name"],信息描述:waitnode["Description"],建模路径:waitnode["Path"],元素属性:waitnode["Property"],元素类型:waitnode["NType"],
    #         节点2信息:名称：waitnode["Name"],信息描述:waitnode["Description"],建模路径:waitnode["Path"],元素属性:waitnode["Property"],元素类型:waitnode["NType"],
    #         ......
    #     }
    #     模型元素和需求元素之间可能存在的关系集合：
    #     {result}
    # }

    # ============================ 输出 JSON 格式结果 ============================

    # 设定要输出的前 k 条边
    k = 5

    # 将边与得分组合，并按得分降序排序
    edges_with_scores = list(zip(tmp_edges, final_scores))
    edges_with_scores.sort(key=lambda x: x[1], reverse=True)
    top_edges = edges_with_scores[:k]
    case_retrieval_ms = (perf_counter() - case_retrieval_start) * 1000
    # [RQ4 TIMING END] Case-based Retrieval

    retrieval_timing = {
        "graph_context_extraction_ms": graph_context_ms,
        "case_based_retrieval_ms": case_retrieval_ms,
    }
    return system_nodes,top_edges,retrieval_timing
