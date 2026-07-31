# #通过将KG中的信息转换为JSON格式数据，调用jar包生成推理关系集合
import subprocess
import json
import os
import pandas as pd
from config import ONTOLOGY_TTL_PATH, REASONER_JAR_PATH, REASONER_WORK_DIR

def build_reasoner_input(model_node, req_node):
    def clean(node):
        return {
            "id": node["ID"],
            "class": node["NType"],
            "data": {
                "Name": str(node["Name"]).strip(),
                "Description": node["Description"],
                "Path": node["Path"],
                "Property": None if pd.isna(node["Property"]) else node["Property"]
            }
        }

    return {
        "individuals": [
            clean(model_node),
            clean(req_node)
        ],
        "relations": [
            {
                "subject": req_node["ID"],
                "property": "relatesTo",
                "object": model_node["ID"]
            }
        ]
    }

def reasoner(data):
    """
    输入data格式如下所示
    返回result为节点之间可能存在的关系集合
    """
    # Java 代码通过相对路径读取本体文件，因此工作目录与 JAR 路径均由
    # config.py 统一管理，支持环境变量覆盖。
    java_workdir = REASONER_WORK_DIR
    jar_path = REASONER_JAR_PATH
    if not jar_path.is_file():
        raise FileNotFoundError(
            f"Ontology reasoner JAR not found: {jar_path}. "
            "Build it with the Gradle wrapper or set TRACE_REASONER_JAR_PATH."
        )
    if not java_workdir.is_dir():
        raise FileNotFoundError(
            f"Ontology reasoner working directory not found: {java_workdir}"
        )
    if not ONTOLOGY_TTL_PATH.is_file():
        raise FileNotFoundError(
            f"Ontology file not found: {ONTOLOGY_TTL_PATH}. "
            "Set TRACE_ONTOLOGY_PATH to override it."
        )

    # 2️⃣ 构造要传给 Java_推理机 的 JSON 数据
    # data = {
    #     "individuals": [
    #         {"id": "ANode1", "class": "Activity", "data": {"Name": "Blow Cooled Air Back to Cabin"}},
    #         {"id": "FRNode1", "class": "FRNode", "data": {"Name": "Cooling"}}
    #     ],
    #     "relations": [
    #         {"subject": "FRNode1", "property": "relatesTo", "object": "ANode1"}
    #     ]
    # }

    json_str = json.dumps(data)

    # # 3️⃣ 调用 jar 并将 JSON 通过 stdin 传入
    process_env = os.environ.copy()
    process_env["TRACE_ONTOLOGY_PATH"] = str(ONTOLOGY_TTL_PATH)
    process = subprocess.Popen(
        ["java", "-jar", str(jar_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # 用文本模式处理输入输出
        cwd=str(java_workdir),  # 关键，指定工作目录
        env=process_env,
    )

    # # 发送 JSON，并等待执行完成
    stdout, stderr = process.communicate(input=json_str)
    #
    # # 4️⃣ 输出结果
    if process.returncode == 0:
        print("================推理机推理完成=============================")
        # print(stdout)
        # 如果想解析为 Python 对象：
    else:
        print("Java 推理失败！")
        print(stderr)
    print(stdout)
    result = json.loads(stdout)
    return result

if __name__ == "__main__":
    data = {
        "individuals": [
            {"id": "ANode1", "class": "Action", "data": {"Name": "Blow Cooled Air Back to Cabin"}},
            {"id": "FRNode1", "class": "Functional_Requirement", "data": {"Name": "Cooling"}}
        ],
        "relations": [
            {"subject": "FRNode1", "property": "relatesTo", "object": "ANode1"}
        ]
    }
    result=reasoner(data)
    print(result)
