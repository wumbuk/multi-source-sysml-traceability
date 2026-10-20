#定义对于知识图谱KG的检索规则
# **维度1**：结构相似性
#
# 与进行关系判断的一对节点存在下列情况：
#
# （1）属于同一子系统的节点及其边（Path相同），#提供子系统功能
#
# （2）与这对节点的类型NType值相同，#用于检索追溯关系（边）
#
# 使用node2vec计算相似度作为打分**struct**
#
# **维度2**：语义相似性
#使用sentence-transformer计算节点名称余弦相似度的分数
#然后把待判模型节点和需求节点的描述先送到一个小模型中，如果可能建立关系，让它给出为什么可以建立关系；如果没有，则让它输出为什么不能建立关系
# **维度3：**对于边关系的筛选：
#从第一个检索规则中得来的节点进行追溯关系的检索
# - SysML 原生结构边（最高可信）
# - 专家标注/历史追溯边（次高）
# - LLM 推理边（可用但可信度低）
#
# 使用边属性Source_type和Confidence进行计分**conf**
#
# 然后对每条证据计算综合得分：
#
# **Score=w1*struct+w2*sem+w3*conf**
#
# 筛选出top-k将信息作为prompt提供给LLM。

#维度1，结构相似性打分工具
def retrieve_node(node,Graph):
    """
    node:待判节点
    nodes:KG节点集合
    return:Get_Nodes是一个和node具有相同NType的检索节点集合
          :System_Nodes是一个和node具有相同Path的检索节点集合
    """
    Get_Nodes= []
    System_Nodes = []
    for _,n in Graph.nodes(data=True):
        if node["NType"]==n["NType"] and node is not n:
            Get_Nodes.append(n)
        if node["Path"]==n["Path"] and node is not n:
            System_Nodes.append(n)
    return Get_Nodes,System_Nodes

import torch
from sentence_transformers.util import cos_sim

def mean_pooling(model_output, attention_mask):
    """对模型输出的 token 嵌入进行平均池化"""
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#维度2，语义相似性打分
def semantic_score(model,model_node,Get_Nodes):
    """
    node:待判节点
    Get_Nodes:检索得到的具有相同NType节点集合
    :return:节点名称相似性打分，追溯缘由
    """
    # model=SentenceTransformer(local_model_path)
    # global answer
    Node_score_dict={}
    emb1=model.encode(model_node["Name"],convert_to_tensor=True, normalize_embeddings=True)
    # try:
    #     answer = model_client.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.5, max_tokens=256)
    #     print(answer)
    #     # Node_score.append(sc1*0.5+sc2*0.5)
    # except Exception as e:
    #     print(e)
    # 遍历节点并构建字典
    for n in Get_Nodes:
        emb2 = model.encode(n["Name"], convert_to_tensor=True, normalize_embeddings=True)
        sim = cos_sim(emb1, emb2)
        # 使用节点的ID作为键，sim作为值
        #在property上，也可以加入一些权重，适当提高这样的相似性分数，
        if n["Property"]==model_node["Property"]:
            sim*=1.5
        Node_score_dict[n["ID"]] = sim.item() if hasattr(sim, 'item') else sim

    return Node_score_dict


#维度3，对于边关系的筛选,依次获取各个边的分数
def edge_score(edges):
    """
    edges:证据节点的边集合
    :return:score,边关系分数(直接提取)
    """
    scores=[]
    for u, v, data in edges:
        scores.append(data["Confidence"])
    return scores

#根据id获取节点
def getnode_byid(Graph,id):
    for _,n in Graph.nodes(data=True):
        if n["ID"] == id:
            return n
    return "节点不存在"

#检索和req_node具有相同NType，并且以tmp_nodes中的节点作为源ID的边集合，以及需求节点集合
def retrieve_edge(tmp_nodes,req_node,Graph):
    """
    :param tmp_nodes: 找到的和模型元素节点类型相同的节点集合
    :param req_node: 需求节点
    :return:选取的边列表tmp_edges,和req_node具有相同NType的节点集合req_nodes
    """
    tmp_edges=[] #模型元素节点和需求节点，相对应Source节点，target节点具有相同NType的边集合
    req_nodes=[] #存储和req_node具有相同NType的节点集合
    seen_ids = set()

    tmp_ids = {n["ID"] for n in tmp_nodes}

    for src, tgt, data in Graph.edges(data=True):
        target = Graph.nodes[tgt]

        if src in tmp_ids and target["NType"] == req_node["NType"]:
            tmp_edges.append(data)

            if target["ID"] not in seen_ids:
                req_nodes.append(target)
                seen_ids.add(target["ID"])

    return tmp_edges,req_nodes

#计算和需求节点的相似性分数
def req_semantic_score(model,req_node,req_nodes):
    emb1=model.encode(req_node["Name"],convert_to_tensor=True, normalize_embeddings=True)
    req_sim_score={}
    for n in req_nodes:
        emb2 = model.encode(n["Name"], convert_to_tensor=True, normalize_embeddings=True)
        sim = cos_sim(emb1, emb2)
        # 使用节点的ID作为键，sim作为值
        req_sim_score[n["ID"]] = sim.item() if hasattr(sim, 'item') else sim

    return req_sim_score

#进行节点和加权
def edge_final_score(mod_sim_score,req_sim_score,tmp_edges):
    final_scores=[]
    for e in tmp_edges:
        source = e["Source"]
        target = e["Target"]
        #=======================这里可能要更改成加权分数========================
        final_score=mod_sim_score[source]+req_sim_score[target]
        if e["Source_type"]=="SysML":
            final_score=final_score*1.0
        elif e["Source_type"]=="LLM":
            final_score=final_score*0.5
        final_scores.append(final_score)
    return final_scores
