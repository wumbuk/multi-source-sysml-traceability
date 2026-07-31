from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


XMI_NS = "http://www.omg.org/spec/XMI/20131001"
DEFAULT_PACKAGE_PATH = [
    "1 Problem Domain",
    "2 White Box",
    "2 Conceptual Subsystems Communication",
]


@dataclass
class PortInfo:
    port_id: str
    name: str
    owner_class_id: str
    element: ET.Element


@dataclass
class ClassInfo:
    class_id: str
    name: str
    element: ET.Element
    ports: list[PortInfo] = field(default_factory=list)


@dataclass
class ConnectorEndInfo:
    role: str
    element: ET.Element


@dataclass
class ConnectorInfo:
    connector_id: str
    element: ET.Element
    ends: list[ConnectorEndInfo] = field(default_factory=list)


@dataclass
class InformationFlowInfo:
    flow_id: str
    name: str
    source_id: str | None
    target_id: str | None
    connector_ids: list[str]
    element: ET.Element


@dataclass
class ParsedModel:
    classes: dict[str, ClassInfo]
    ports: dict[str, PortInfo]
    connectors: dict[str, ConnectorInfo]
    flows: list[InformationFlowInfo]


class XMIInformationExtractor:
    """按可配置策略提取端口追溯相关XMI片段。"""

    def parse(self, xmi_path: str | Path, package_path: list[str] | None = None) -> ParsedModel:
        tree = ET.parse(xmi_path)
        root = tree.getroot()
        scope_root = find_scope_root(root, package_path or DEFAULT_PACKAGE_PATH)

        if scope_root is None:
            joined = " / ".join(package_path or DEFAULT_PACKAGE_PATH)
            raise ValueError(f"未找到指定Package路径: {joined}")

        classes: dict[str, ClassInfo] = {}
        ports: dict[str, PortInfo] = {}
        connectors: dict[str, ConnectorInfo] = {}
        flows: list[InformationFlowInfo] = []

        for elem in scope_root.iter():
            if local_name(elem.tag) == "packagedElement" and attr(elem, "type") == "uml:Class":
                class_id = attr(elem, "id")
                class_name = elem.attrib.get("name")
                if not class_id or not class_name:
                    continue

                class_info = ClassInfo(class_id=class_id, name=class_name, element=elem)

                for child in elem:
                    if local_name(child.tag) != "ownedAttribute":
                        continue
                    if attr(child, "type") != "uml:Port":
                        continue
                    port_id = attr(child, "id")
                    port_name = child.attrib.get("name")
                    if not port_id or not port_name:
                        continue
                    port_info = PortInfo(
                        port_id=port_id,
                        name=port_name,
                        owner_class_id=class_id,
                        element=child,
                    )
                    class_info.ports.append(port_info)
                    ports[port_id] = port_info

                classes[class_id] = class_info

            if local_name(elem.tag) == "ownedConnector" and attr(elem, "type") == "uml:Connector":
                connector_id = attr(elem, "id")
                if not connector_id:
                    continue
                connector = ConnectorInfo(connector_id=connector_id, element=elem)
                for child in elem:
                    if local_name(child.tag) != "end":
                        continue
                    role = child.attrib.get("role")
                    if role:
                        connector.ends.append(ConnectorEndInfo(role=role, element=child))
                connectors[connector_id] = connector

            if local_name(elem.tag) == "packagedElement" and attr(elem, "type") == "uml:InformationFlow":
                flow_id = attr(elem, "id")
                if not flow_id:
                    continue
                flow_name = elem.attrib.get("name", "")
                source_id = None
                target_id = None
                connector_ids: list[str] = []

                for child in elem:
                    c_name = local_name(child.tag)
                    if c_name == "informationSource":
                        source_id = child.attrib.get(f"{{{XMI_NS}}}idref")
                    elif c_name == "informationTarget":
                        target_id = child.attrib.get(f"{{{XMI_NS}}}idref")
                    elif c_name == "realizingConnector":
                        conn_id = child.attrib.get(f"{{{XMI_NS}}}idref")
                        if conn_id:
                            connector_ids.append(conn_id)

                flows.append(
                    InformationFlowInfo(
                        flow_id=flow_id,
                        name=flow_name,
                        source_id=source_id,
                        target_id=target_id,
                        connector_ids=connector_ids,
                        element=elem,
                    )
                )

        return ParsedModel(classes=classes, ports=ports, connectors=connectors, flows=flows)

    def select_systems(
        self,
        model: ParsedModel,
        include_system_names: list[str] | None = None,
        anchor_system_name: str | None = None,
        hop: int = 1,
    ) -> set[str]:
        """
        选择目标子系统。

        1) include_system_names 非空时：按名称精确匹配。
        2) 否则 anchor_system_name 非空时：以锚点系统为中心，根据InformationFlow做hop跳扩展。
        3) 都为空时：默认选择所有包含端口的Class。
        """
        name_to_id = {
            normalize_name(c.name): c.class_id
            for c in model.classes.values()
        }

        if include_system_names:
            return {
                name_to_id[normalize_name(name)]
                for name in include_system_names
                if normalize_name(name) in name_to_id
            }

        anchor_key = normalize_name(anchor_system_name) if anchor_system_name else None
        if anchor_key and anchor_key in name_to_id:
            selected = {name_to_id[anchor_key]}
            for _ in range(max(1, hop)):
                next_ids = set(selected)
                for flow in model.flows:
                    if flow.source_id in selected and flow.target_id:
                        next_ids.add(flow.target_id)
                    if flow.target_id in selected and flow.source_id:
                        next_ids.add(flow.source_id)
                selected = next_ids
            return selected

        return {
            class_info.class_id
            for class_info in model.classes.values()
            if class_info.ports
        }

    def collect_fragments(
        self,
        model: ParsedModel,
        selected_system_ids: set[str],
    ) -> tuple[list[ClassInfo], list[ConnectorInfo], list[InformationFlowInfo]]:
        # 第一部分：目标子系统及其端口
        selected_classes = [
            c for c in model.classes.values()
            if c.class_id in selected_system_ids and c.ports
        ]
        selected_classes.sort(key=lambda x: x.name)

        selected_port_ids = {
            p.port_id
            for c in selected_classes
            for p in c.ports
        }

        # 第二部分：只保留两端都落在目标端口集合中的Connector
        selected_connectors: list[ConnectorInfo] = []
        selected_connector_ids: set[str] = set()
        for conn in model.connectors.values():
            roles = [e.role for e in conn.ends]
            if roles and all(role in selected_port_ids for role in roles):
                selected_connectors.append(conn)
                selected_connector_ids.add(conn.connector_id)

        selected_connectors.sort(key=lambda x: x.connector_id)

        # 第三部分：只保留带realizingConnector且与已选Connector相关的信息流
        selected_flows: list[InformationFlowInfo] = []
        for flow in model.flows:
            if not flow.connector_ids:
                continue
            by_connector = any(cid in selected_connector_ids for cid in flow.connector_ids)
            if by_connector:
                selected_flows.append(flow)

        selected_flows.sort(key=lambda x: x.flow_id)
        return selected_classes, selected_connectors, selected_flows

    def build_output_text(
        self,
        classes: Iterable[ClassInfo],
        connectors: Iterable[ConnectorInfo],
        flows: Iterable[InformationFlowInfo],
    ) -> str:
        lines: list[str] = []

        lines.append("#====================================================（第一部分）定义子系统以及端口（xmi:id作为唯一标识）========================================================")
        for class_info in classes:
            lines.extend(format_class_fragment(class_info))

        lines.append("")
        lines.append("#========================================（第二部分）这些是Connector，也就是连接线，这里的每个Connector下的两个role分别是两个端口========================================")
        for conn in connectors:
            lines.extend(format_connector_fragment(conn))

        lines.append("")
        lines.append("#======================================（第三部分）子系统之间的信息流动==========================================")
        for flow in flows:
            lines.extend(format_information_flow_fragment(flow))

        return "\n".join(lines).strip() + "\n"

    def extract(
        self,
        xmi_path: str | Path,
        output_path: str | Path,
        include_system_names: list[str] | None = None,
        anchor_system_name: str | None = None,
        hop: int = 1,
        package_path: list[str] | None = None,
    ) -> Path:
        model = self.parse(xmi_path, package_path=package_path)
        selected_ids = self.select_systems(
            model,
            include_system_names=include_system_names,
            anchor_system_name=anchor_system_name,
            hop=hop,
        )
        classes, connectors, flows = self.collect_fragments(model, selected_ids)
        text = self.build_output_text(classes, connectors, flows)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        return output


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def normalize_name(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().split())


def attr(elem: ET.Element, short_name: str) -> str | None:
    # 兼容 xmi:id / xmi:type / 普通属性
    xmi_key = f"{{{XMI_NS}}}{short_name}"
    if xmi_key in elem.attrib:
        return elem.attrib[xmi_key]
    if short_name in elem.attrib:
        return elem.attrib[short_name]
    return None


def is_packaged_package(elem: ET.Element, expected_name: str | None = None) -> bool:
    if local_name(elem.tag) != "packagedElement":
        return False
    if attr(elem, "type") != "uml:Package":
        return False
    if expected_name is not None and elem.attrib.get("name") != expected_name:
        return False
    return True


def find_scope_root(root: ET.Element, package_path: list[str]) -> ET.Element | None:
    if not package_path:
        return root

    def walk_find_first(node: ET.Element, name: str) -> Iterable[ET.Element]:
        for e in node.iter():
            if is_packaged_package(e, name):
                yield e

    def walk_match_path(node: ET.Element, names: list[str], idx: int) -> ET.Element | None:
        if idx >= len(names):
            return node
        for child in node:
            if is_packaged_package(child, names[idx]):
                found = walk_match_path(child, names, idx + 1)
                if found is not None:
                    return found
            else:
                # 有些模型中中间会有容器节点，继续下探匹配同一层目标
                found = walk_match_path(child, names, idx)
                if found is not None:
                    return found
        return None

    for start in walk_find_first(root, package_path[0]):
        if len(package_path) == 1:
            return start
        matched = walk_match_path(start, package_path, 1)
        if matched is not None:
            return matched
    return None


def ordered_attrs(src: dict[str, str], keys: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    used = set()
    for key in keys:
        if key in src:
            out.append((key, src[key]))
            used.add(key)
    for key, value in src.items():
        if key in used:
            continue
        out.append((key, value))
    return out


def to_plain_attrs(elem: ET.Element) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in elem.attrib.items():
        if key.startswith("{"):
            if key.startswith(f"{{{XMI_NS}}}"):
                out[f"xmi:{key.split('}', 1)[1]}"] = value
            else:
                out[key.split("}", 1)[1]] = value
        else:
            out[key] = value
    return out


def format_attrs(attrs: list[tuple[str, str]]) -> str:
    return " ".join(f'{k}="{v}"' for k, v in attrs)


def format_class_fragment(class_info: ClassInfo) -> list[str]:
    lines: list[str] = []
    class_attrs = to_plain_attrs(class_info.element)
    class_line_attrs = ordered_attrs(class_attrs, ["xmi:type", "xmi:id", "name"])

    if class_info.ports:
        lines.append(f"<packagedElement {format_attrs(class_line_attrs)}>")
        for port in class_info.ports:
            port_attrs = to_plain_attrs(port.element)
            port_line_attrs = ordered_attrs(
                port_attrs,
                ["xmi:type", "xmi:id", "name", "visibility", "aggregation", "isConjugated", "type"],
            )
            lines.append(f"    <ownedAttribute {format_attrs(port_line_attrs)}/>")
        lines.append("</packagedElement>")
    else:
        lines.append(f"<packagedElement {format_attrs(class_line_attrs)}/>")

    return lines


def format_connector_fragment(conn: ConnectorInfo) -> list[str]:
    lines: list[str] = []
    conn_attrs = to_plain_attrs(conn.element)
    conn_line_attrs = ordered_attrs(conn_attrs, ["xmi:type", "xmi:id", "visibility"])
    lines.append(f"<ownedConnector {format_attrs(conn_line_attrs)}>")

    for end in conn.ends:
        end_attrs = to_plain_attrs(end.element)
        end_line_attrs = ordered_attrs(
            end_attrs,
            ["xmi:type", "xmi:id", "partWithPort", "role"],
        )
        lines.append(f"    <end {format_attrs(end_line_attrs)}/>")

    lines.append("</ownedConnector>")
    return lines


def format_information_flow_fragment(flow: InformationFlowInfo) -> list[str]:
    lines: list[str] = []
    flow_attrs = to_plain_attrs(flow.element)
    flow_line_attrs = ordered_attrs(flow_attrs, ["xmi:type", "xmi:id", "name"])
    lines.append(f"<packagedElement {format_attrs(flow_line_attrs)}>")

    for child in flow.element:
        c_name = local_name(child.tag)
        if c_name not in {
            "conveyed",
            "informationSource",
            "informationTarget",
            "realizingActivityEdge",
            "realizingConnector",
        }:
            continue
        c_attrs = to_plain_attrs(child)
        c_line_attrs = ordered_attrs(c_attrs, ["xmi:idref"])
        lines.append(f"    <{c_name} {format_attrs(c_line_attrs)}/>")

    lines.append("</packagedElement>")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取端口追溯相关XMI片段")
    parser.add_argument("--xmi", required=True, help="输入XMI文件路径")
    parser.add_argument("--output", required=True, help="输出TXT路径")
    parser.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="指定系统名称列表（精确匹配），例如: --systems VCCU Cooling Heating",
    )
    parser.add_argument(
        "--anchor",
        default=None,
        help="锚点系统名称（当不提供--systems时使用）",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=1,
        help="锚点扩展跳数，默认1",
    )
    parser.add_argument(
        "--package-path",
        nargs="*",
        default=DEFAULT_PACKAGE_PATH,
        help=(
            "限定解析的Package层级路径，默认: "
            "1 Problem Domain -> 2 White Box -> 2 Conceptual Subsystems Communication"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extractor = XMIInformationExtractor()
    output = extractor.extract(
        xmi_path=args.xmi,
        output_path=args.output,
        include_system_names=args.systems,
        anchor_system_name=args.anchor,
        hop=args.hop,
        package_path=args.package_path,
    )
    print(f"提取完成: {output}")


if __name__ == "__main__":
    main()
