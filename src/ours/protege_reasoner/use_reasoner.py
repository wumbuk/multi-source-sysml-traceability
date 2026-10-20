# #通过将KG中的信息转换为JSON格式数据，调用jar包生成推理关系集合
import subprocess
import json
import os
import shlex
import re
from functools import lru_cache
import pandas as pd
from config import ONTOLOGY_TTL_PATH, REASONER_JAR_PATH, REASONER_WORK_DIR


@lru_cache(maxsize=1)
def _get_java_args():
    configured_args = os.getenv("TRACE_REASONER_JAVA_ARGS")
    if configured_args is not None:
        return shlex.split(configured_args)

    version_process = subprocess.run(
        ["java", "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    version_output = f"{version_process.stdout}\n{version_process.stderr}"
    match = re.search(r'version "(?:1\.)?(\d+)', version_output)
    if match and int(match.group(1)) >= 9:
        return ["--add-opens=java.base/java.lang=ALL-UNNAMED"]
    return []

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
    java_args = _get_java_args()
    java_command = ["java", *java_args, "-jar", str(jar_path)]
    process = subprocess.run(
        java_command,
        input=json_str,
        capture_output=True,
        text=True,
        cwd=str(java_workdir),
        env=process_env,
        check=False,
    )

    stdout = process.stdout.strip()
    stderr = process.stderr.strip()
    if process.returncode != 0:
        raise RuntimeError(
            "Java ontology reasoner failed "
            f"(exit code {process.returncode}).\n"
            f"Command: {shlex.join(java_command)}\n"
            f"stderr:\n{stderr or '<empty>'}"
        )
    if not stdout:
        raise RuntimeError(
            "Java ontology reasoner returned empty stdout.\n"
            f"stderr:\n{stderr or '<empty>'}"
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Java ontology reasoner returned invalid JSON.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr or '<empty>'}"
        ) from exc

    print("================推理机推理完成=============================")
    print(stdout)
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
