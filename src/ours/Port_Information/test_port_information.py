import re
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

from config import PORT_INFORMATION_PATH


# ====================== 1. 读取文件 ======================
def read_uml_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    try:
        ET.fromstring(content)
        return content
    except ET.ParseError:
        # The bundled legacy file contains prose and three sets of XML fragments,
        # with unclosed Class elements. Only normalize this documented format;
        # malformed standalone XML must still raise an error.
        if not all(label in content for label in ("第一部分", "第二部分", "第三部分")):
            raise
    return normalize_legacy_fragments(content)


def normalize_legacy_fragments(content: str) -> str:
    """Wrap the bundled annotated fragments without inventing missing relations."""
    parts = [
        '<root xmlns:xmi="http://www.omg.org/spec/XMI/20131001" '
        'xmlns:uml="http://www.omg.org/spec/UML/20131001">'
    ]
    class_open = False
    class_count = 0
    tokens = re.findall(
        r'<packagedElement\b[^>]*\bxmi:type="uml:Class"[^>]*>'
        r'|<ownedAttribute\b[^>]*\bxmi:type="uml:Port"[^>]*/>',
        content,
    )
    for token in tokens:
        if token.startswith("<packagedElement"):
            if class_open:
                parts.append("</packagedElement>")
            parts.append(token)
            class_open = not token.rstrip().endswith("/>")
            class_count += 1
        elif class_open:
            parts.append(token)
        else:
            raise ValueError("Legacy XMI contains a port without an owning class.")
    if class_open:
        parts.append("</packagedElement>")
    connectors = re.findall(
        r'<ownedConnector\b[^>]*>.*?</ownedConnector>', content, re.S
    )
    flows = re.findall(
        r'<packagedElement\b[^>]*\bxmi:type="uml:InformationFlow"[^>]*>'
        r'.*?</packagedElement>', content, re.S
    )
    if not class_count or not connectors or not flows:
        raise ValueError("Legacy XMI must include classes, connectors and information flows.")
    parts.extend(connectors)
    parts.extend(flows)
    parts.append("</root>")
    normalized = "\n".join(parts)
    ET.fromstring(normalized)
    return normalized


# ====================== 2. 正则提取属性 ======================
def extract_attr(line: str, attr: str) -> str | None:
    pattern = rf'{attr}="([^"]+)"'
    match = re.search(pattern, line)
    return match.group(1) if match else None


def local_name(name):
    return name.rsplit('}', 1)[-1]


def xmi_attribute(element, attribute):
    return next((value for name, value in element.attrib.items()
                 if name.startswith('{') and local_name(name) == attribute), None)


# ====================== 3. 解析子系统与端口 ======================
def parse_subsystems_and_ports(content: str) -> tuple[dict, dict]:
    subs = {}
    ports = {}

    def visit(element, owner_id=None):
        element_type = (xmi_attribute(element, 'type') or '').rsplit(':', 1)[-1]
        if element_type == 'Class':
            owner_id = xmi_attribute(element, 'id')
            if owner_id and element.get('name'):
                subs[owner_id] = element.get('name')
        elif element_type == 'Port' and owner_id in subs:
            port_id = xmi_attribute(element, 'id')
            port_name = element.get('name')
            if port_id and port_name:
                ports[port_id] = {
                    'port_name': port_name.strip(),
                    'sub_id': owner_id,
                    'sub_name': subs[owner_id]
                }
        for child in element:
            visit(child, owner_id)

    visit(ET.fromstring(content))
    return subs, ports


# ====================== 4. 解析 Connector ======================
def parse_connectors(content: str) -> dict:
    connectors = {}
    for element in ET.fromstring(content).iter():
        if (xmi_attribute(element, 'type') or '').rsplit(':', 1)[-1] != 'Connector':
            continue
        conn_id = xmi_attribute(element, 'id')
        if conn_id:
            connectors[conn_id] = [
                child.get('role') for child in element
                if local_name(child.tag) == 'end' and child.get('role')
            ]
    return connectors


# ====================== 5. 解析信息流 ======================
def parse_information_flows(content: str) -> list[dict]:
    flows = []
    for element in ET.fromstring(content).iter():
        if (xmi_attribute(element, 'type') or '').rsplit(':', 1)[-1] != 'InformationFlow':
            continue
        references = {
            local_name(child.tag): xmi_attribute(child, 'idref') for child in element
        }
        flows.append({
            'flow_name': element.get('name'),
            'source': references.get('informationSource'),
            'target': references.get('informationTarget'),
            'conn_id': references.get('realizingConnector')
        })
    return flows


# ====================== 6. 计算端口方向（保留但不输出） ======================
def compute_port_directions(flows, connectors, ports):
    port_dir = defaultdict(lambda: {'in': False, 'out': False})

    for f in flows:
        conn_id = f.get('conn_id')
        source_sub = f.get('source')
        target_sub = f.get('target')

        if not conn_id or len(connectors.get(conn_id, [])) != 2:
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

        if not conn_id or len(connectors.get(conn_id, [])) != 2:
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

        if ports[dst_port]['sub_id'] != target_sub_id:
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
    if PORT_INFORMATION_PATH is None:
        raise SystemExit("Port information is disabled by configuration.")
    parse_port_information(
        str(PORT_INFORMATION_PATH)
    )
