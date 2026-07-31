import re
import json
from collections import defaultdict

from config import PORT_INFORMATION_PATH


# ====================== 1. 读取文件 ======================
def read_uml_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# ====================== 2. 正则提取属性 ======================
def extract_attr(line: str, attr: str) -> str | None:
    pattern = rf'{attr}="([^"]+)"'
    match = re.search(pattern, line)
    return match.group(1) if match else None


# ====================== 3. 解析子系统与端口 ======================
def parse_subsystems_and_ports(content: str) -> tuple[dict, dict]:
    subs = {}
    ports = {}
    current_sub_id = None
    current_sub_name = None

    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if '<packagedElement' in line and 'uml:Class' in line:
            sub_id = extract_attr(line, 'xmi:id')
            sub_name = extract_attr(line, 'name')
            if sub_id and sub_name:
                subs[sub_id] = sub_name
                current_sub_id = sub_id
                current_sub_name = sub_name

        elif '<ownedAttribute' in line and 'uml:Port' in line:
            port_id = extract_attr(line, 'xmi:id')
            port_name = extract_attr(line, 'name')
            if port_id and port_name and current_sub_id:
                ports[port_id] = {
                    'port_name': port_name.strip(),
                    'sub_id': current_sub_id,
                    'sub_name': current_sub_name
                }

    return subs, ports


# ====================== 4. 解析 Connector ======================
def parse_connectors(content: str) -> dict:
    connectors = {}
    current_conn = None

    lines = content.splitlines()
    for line in lines:
        line = line.strip()
        if '<ownedConnector' in line and 'uml:Connector' in line:
            conn_id = extract_attr(line, 'xmi:id')
            if conn_id:
                current_conn = conn_id
                connectors[current_conn] = []

        elif '<end' in line and 'role=' in line and current_conn:
            role = extract_attr(line, 'role')
            if role:
                connectors[current_conn].append(role)

    return connectors


# ====================== 5. 解析信息流 ======================
def parse_information_flows(content: str) -> list[dict]:
    flows = []
    flow = {}
    in_flow = False

    lines = content.splitlines()
    for line in lines:
        line = line.strip()

        if '<packagedElement' in line and 'uml:InformationFlow' in line:
            in_flow = True
            flow = {
                'flow_name': extract_attr(line, 'name'),
                'source': None,
                'target': None,
                'conn_id': None
            }

        elif '</packagedElement>' in line and in_flow:
            in_flow = False
            flows.append(flow)

        elif in_flow:
            if 'informationSource' in line:
                flow['source'] = extract_attr(line, 'xmi:idref')
            elif 'informationTarget' in line:
                flow['target'] = extract_attr(line, 'xmi:idref')
            elif 'realizingConnector' in line:
                flow['conn_id'] = extract_attr(line, 'xmi:idref')

    return flows


# ====================== 6. 计算端口方向（保留但不输出） ======================
def compute_port_directions(flows, connectors, ports):
    port_dir = defaultdict(lambda: {'in': False, 'out': False})

    for f in flows:
        conn_id = f.get('conn_id')
        source_sub = f.get('source')
        target_sub = f.get('target')

        if not conn_id or conn_id not in connectors:
            continue
        if not source_sub or not target_sub:
            continue

        p1, p2 = connectors[conn_id]
        for pid in [p1, p2]:
            if pid not in ports:
                continue
            sub = ports[pid]['sub_id']
            if sub == source_sub:
                port_dir[pid]['out'] = True
            if sub == target_sub:
                port_dir[pid]['in'] = True

    port_final_dir = {}
    for pid, d in port_dir.items():
        if d['in'] and d['out']:
            port_final_dir[pid] = 'inout'
        elif d['out']:
            port_final_dir[pid] = 'out'
        elif d['in']:
            port_final_dir[pid] = 'in'
        else:
            port_final_dir[pid] = ''
    return port_final_dir


# ====================== 7. 构建信息流 JSON（核心修改） ======================
def build_all_flows_json(subs, ports, connectors, flows, port_dir):
    all_flows = []

    for f in flows:
        conn_id = f.get('conn_id')
        source_sub_id = f.get('source')
        target_sub_id = f.get('target')

        if not conn_id or conn_id not in connectors:
            continue
        if not source_sub_id or not target_sub_id:
            continue

        p_a, p_b = connectors[conn_id]
        if p_a not in ports or p_b not in ports:
            continue

        src_port, dst_port = None, None
        if ports[p_a]['sub_id'] == source_sub_id:
            src_port = p_a
            dst_port = p_b
        elif ports[p_b]['sub_id'] == source_sub_id:
            src_port = p_b
            dst_port = p_a
        else:
            continue

        # ✅ 端口名称（不带 in/out）
        src_name = ports[src_port]['port_name']
        dst_name = ports[dst_port]['port_name']

        # ✅ 提取语义信息（核心增强）
        flow_name = f["flow_name"] or ""
        semantic_info = flow_name.replace("flow for ", "").strip()

        flow_item = {
            "信息流名称": flow_name,
            "流动信息": semantic_info,

            "源系统": subs[source_sub_id],
            "信息输出端口": src_name,
            "信息输出端口ID": src_port,

            "目标系统": subs[target_sub_id],
            "信息接收端口": dst_name,
            "信息接收端口ID": dst_port,

            "连接器ID": conn_id
        }

        all_flows.append(flow_item)

    return json.dumps(all_flows, ensure_ascii=False, indent=2)


# ====================== 主函数 ======================
def parse_port_information(port_information_path):
    content = read_uml_file(port_information_path)

    subs, ports = parse_subsystems_and_ports(content)
    connectors = parse_connectors(content)
    flows = parse_information_flows(content)
    port_dir = compute_port_directions(flows, connectors, ports)

    result_json = build_all_flows_json(subs, ports, connectors, flows, port_dir)

    print(result_json)
    return result_json


# ====================== 运行 ======================
if __name__ == '__main__':
    parse_port_information(
        str(PORT_INFORMATION_PATH)
    )
