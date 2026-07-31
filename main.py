import atexit
import sys
import traceback
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from sentence_transformers import SentenceTransformer
import json
from config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME_OR_PATH,
    JUDGE_CSV_PATH,
    KG_CSV_PATH,
    OUTPUT_RESULT_PATH,
    PORT_INFORMATION_PATH,
    SEMANTIC_DATASET_PATH,
)

#==============进行元素功能的扩写==============
from enlarge_function.Get_Model_Function import enlarge_element

#==============进行知识图谱初始化===============
from kg_retrieve.construct_KG import construct_kg_init

#==============KG可视化====================
from kg_retrieve.visualize_kg import visualize_kg

#==============构建待判节点对集合================
from kg_retrieve.read_sysml_raw_data import build_node_pairs,add_nodes_from_pairs,update_subsystem_nodes,update_subsystem_edges

#==============知识图谱检索=================
from kg_retrieve.use_rules import retrieve

#===============推理机======================
from protege_reasoner.use_reasoner import reasoner,build_reasoner_input

#===============构建json格式信息================
from construct_json import construct_inf_json

#===============进行LLM的调用===================
from call_llm.call_llm import call_qwen_two_stage

#===============更新KG信息======================
from add_information_to_graph.add_information import add_edges

#===============解析端口信息===================
from Port_Information.test_port_information import parse_port_information

#===============数据划分=======================
from Data.Data_Divide import split_dataset

#=======================运行参数==========================

port_information_path = str(PORT_INFORMATION_PATH)
output_result_path = OUTPUT_RESULT_PATH

# 1. 加载模型和分词器（使用 sentence-transformers 的模型名）
model = SentenceTransformer(
    EMBEDDING_MODEL_NAME_OR_PATH,
    device=EMBEDDING_DEVICE,
)



# ========判断是否为端口类型样本 ==============
def is_port_node(node):
    return node.get("NType") in [
        "Information_port",
        "Control_port",
        "physical_port"
    ]


def resolve_result_paths(configured_path):
    """同时支持结果目录和具体 JSON 文件路径。"""
    output_path = Path(configured_path)
    if output_path.suffix.lower() == ".json":
        result_path = output_path
        timing_path = output_path.with_name(
            f"{output_path.stem}_timing.json"
        )
    else:
        result_path = output_path / "predictions.json"
        timing_path = output_path / "timing.json"

    result_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    return result_path, timing_path


def save_progress(result_path, timing_path, results, timings, offline_timing):
    """逐样本保存检查点，避免长时间实验在结束阶段丢失结果。"""
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with timing_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "unit": "ms",
                "offline_timing": offline_timing,
                "samples": timings,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_emergency_results(state, exception_info=None, reason="abnormal_exit"):
    """将异常退出前的结果保存到项目根目录的临时文件夹。"""
    if state["saved"] or state["completed"]:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temporary_dir = (
        Path(__file__).resolve().parent
        / "temporary_results"
        / timestamp
    )

    try:
        temporary_dir.mkdir(parents=True, exist_ok=False)
        save_progress(
            temporary_dir / "predictions.json",
            temporary_dir / "timing.json",
            state["results"],
            state["timings"],
            state["offline_timing"],
        )

        error_data = {
            "reason": reason,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "completed_sample_total": len(state["results"]),
        }
        if exception_info is not None:
            exc_type, exc_value, exc_traceback = exception_info
            error_data.update({
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_value),
                "traceback": "".join(
                    traceback.format_exception(
                        exc_type,
                        exc_value,
                        exc_traceback,
                    )
                ),
            })

        with (temporary_dir / "error.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

        state["saved"] = True
        state["temporary_dir"] = temporary_dir
        print(
            f"\n程序异常退出，临时结果已保存到: {temporary_dir}",
            file=sys.stderr,
        )
    except Exception as save_error:
        print(
            f"\n异常结果保存失败: {save_error}",
            file=sys.stderr,
        )


def install_emergency_save(results, timings, offline_timing):
    """注册未捕获异常和解释器退出时的结果保存处理器。"""
    state = {
        "results": results,
        "timings": timings,
        "offline_timing": offline_timing,
        "completed": False,
        "saved": False,
        "temporary_dir": None,
    }
    original_excepthook = sys.excepthook

    def emergency_excepthook(exc_type, exc_value, exc_traceback):
        save_emergency_results(
            state,
            exception_info=(exc_type, exc_value, exc_traceback),
            reason="unhandled_exception",
        )
        original_excepthook(exc_type, exc_value, exc_traceback)

    def emergency_atexit():
        save_emergency_results(state, reason="interpreter_exit")

    sys.excepthook = emergency_excepthook
    atexit.register(emergency_atexit)
    return state


if __name__ == "__main__":
    result = []
    timing_records = []
    offline_timing = {}
    emergency_state = install_emergency_save(
        result,
        timing_records,
        offline_timing,
    )
    result_output_path, timing_output_path = resolve_result_paths(
        output_result_path
    )
    semantic_dataset_path = SEMANTIC_DATASET_PATH
    # #=================1️⃣构建路径=================================
    data_path = str(JUDGE_CSV_PATH)
    kg_path = str(KG_CSV_PATH)
    #=================Semantic Enrichment=====================
    # [RQ4 OFFLINE TIMING]
    # 语义丰富会先对元素去重，再逐个调用 LLM，因此应报告：
    # 1) 整批离线预处理总耗时；或 2) 每个唯一模型元素的平均耗时。
    # 它不应直接分摊为当前 358 个样本的在线推理耗时。
    # semantic_enrichment_start = perf_counter()
    # enriched_dataset_path = enlarge_element(
    #     mr_file=semantic_dataset_path,
    #     output_path=semantic_dataset_path,
    # )
    # offline_timing["semantic_enrichment_total_ms"] = (
    #     perf_counter() - semantic_enrichment_start
    # ) * 1000

    # enriched_columns = pd.read_csv(enriched_dataset_path, nrows=0).columns
    # if "contextA" not in enriched_columns:
    #     raise RuntimeError(
    #         f"语义丰富结果缺少 contextA 字段: {enriched_dataset_path}"
    #     )

    #=================按 Relation 分层划分训练集和测试集=====================
    # Data_Divide 支持传入目录，并分别生成 KG.csv 和 Judge.csv。
    # data_divide_start = perf_counter()
    # train_path, test_path = split_dataset(
    #     input_path=enriched_dataset_path,
    #     train_output=KG_CSV_PATH,
    #     test_output=JUDGE_CSV_PATH,
    # )
    # offline_timing["data_divide_total_ms"] = (
    #     perf_counter() - data_divide_start
    # ) * 1000

    # 后续流程直接使用本次划分生成的文件。
    train_path = KG_CSV_PATH
    test_path = JUDGE_CSV_PATH
    kg_path = str(train_path)
    data_path = str(test_path)

    #=================初始化知识库并构建节点对=====================
    #这里后续要加的一点是，要把端口之间的交互信息，活动之间的交互信息给保存在知识库中，用于后续的检索
    Graph,counters,ss_counter=construct_kg_init(kg_path)
    judge_df=pd.read_csv(data_path)
    node_pairs,node_index=build_node_pairs(judge_df,Graph=Graph,counters=counters)
    #=================将待判断的节点先加入Graph的节点集合中去================
    add_nodes_from_pairs(Graph,node_pairs)
    # print("nodes:",nodes)
    # for node in nodes:
    #     Graph.add_node(node["ID"], **node)
    # 更新子系统节点
    subsystem_index, ss_counter = update_subsystem_nodes(
        Graph,
        nodes=[pair["A"] for pair in node_pairs] + [pair["B"] for pair in node_pairs],
        ss_counter=ss_counter
    )
    # 建立 belongsToSubsystem 边
    update_subsystem_edges(
        Graph,
        nodes=[pair["A"] for pair in node_pairs] + [pair["B"] for pair in node_pairs],
        subsystem_index=subsystem_index
    )
    print(f"节点数量{len(Graph.nodes)}边数量{len(Graph.edges)}")
    # ============将系统XMI片段作为端口信息port_information================
    port_information = parse_port_information(port_information_path)

    #遍历节点对构建信息
    k=101
    for pair in node_pairs[99:]:
        model_node=pair["A"]
        req_node=pair["B"]
        print(f"=======================第{k}对节点==================================")
        print("节点A:",model_node)
        print("节点B:",req_node)

        # [RQ4 TIMING START] 单样本在线端到端耗时。
        # 不包含终端打印、结果文件写入和离线语义丰富。
        sample_total_start = perf_counter()

        #==========Ontology Reasoning time compute=======================
        # [RQ4 TIMING START] Ontology Reasoning：
        # 包含推理输入构建及本体推理机执行。
        ontology_start = perf_counter()
        data=build_reasoner_input(model_node,req_node)
        relation_set=reasoner(data)
        ontology_reasoning_ms = (perf_counter() - ontology_start) * 1000
        # [RQ4 TIMING END] Ontology Reasoning


        #==========检索案例与上下文,以及端口信息检索=========================
        # retrieve() 内部已分别计时：
        # 1) Graph-based Context Extraction：遍历 KG 得到 tmp_nodes/system_nodes
        # 2) Case-based Retrieval：语义编码、候选边检索、打分与 Top-K
        system_nodes,top_edges,retrieval_timing=retrieve(
            Graph,
            model_node,
            req_node,
            model
        )

        #构建最终的json格式信息
        # 该步骤不属于五个核心组件，单独记录为组装开销，
        # 用于解释 online_total_ms 与核心组件耗时之和的差值。
        information_construction_start = perf_counter()
        information_json=construct_inf_json(Graph,model_node,req_node,top_edges,system_nodes,relation_set)
        information_construction_ms = (
            perf_counter() - information_construction_start
        ) * 1000

        #============判断是否为端口类型数据==================
        # is_port=is_port_node(model_node)
        is_port = False  # 目前暂时不使用端口信息流动，直接传入 False

        #============LLM Inference======================
        # [RQ4 TIMING START] LLM Inference：
        # 覆盖两阶段 API 调用、响应等待和输出解析。
        llm_inference_start = perf_counter()
        answer=call_qwen_two_stage(information_json,is_port,port_information)
        llm_inference_ms = (perf_counter() - llm_inference_start) * 1000
        # [RQ4 TIMING END] LLM Inference
        online_total_ms = (perf_counter() - sample_total_start) * 1000

        # 构建保存时间信息
        timing_records.append({
            "序号": k + 1,
            "NameA": model_node.get("Name", ""),
            "NameB": req_node.get("Name", ""),
            "timing_ms": {
                "graph_context_extraction": retrieval_timing[
                    "graph_context_extraction_ms"
                ],
                "case_based_retrieval": retrieval_timing[
                    "case_based_retrieval_ms"
                ],
                "ontology_reasoning": ontology_reasoning_ms,
                "llm_inference": llm_inference_ms,
                "information_construction_overhead": (
                    information_construction_ms
                ),
                "online_total": online_total_ms,
            }
        })
        print("===============================LLM输出========================")

        #====================将LLM判断的结果更新到KG=========================
        #每条边的包含的信息为
        tmp_edge=add_edges(model_node,req_node,answer)
        print("===============================构建的待添加边=====================")
        #这里有一个要注意的点是，向知识图谱中添加的边的SourceID和TargetID
        print("tmp_edge:",tmp_edge)
        # if tmp_edge:
        #     Graph.add_edge(tmp_edge["Source"], tmp_edge["Target"], **tmp_edge)
        result.append(answer)
        save_progress(
            result_output_path,
            timing_output_path,
            result,
            timing_records,
            offline_timing,
        )

        k+=1
        print()
        print()

    save_progress(
        result_output_path,
        timing_output_path,
        result,
        timing_records,
        offline_timing,
    )
    print(f"Prediction results saved to: {result_output_path}")
    print(f"RQ4 timing results saved to: {timing_output_path}")
    emergency_state["completed"] = True
