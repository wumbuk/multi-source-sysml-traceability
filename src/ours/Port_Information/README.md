# Port Information

English | [中文](README_zh-CN.md)

This component extracts ports and information flows from SysML/XMI models and converts XMI relations into structured JSON for subsequent traceability reasoning.

Processing has two stages:

1. `information_extract.py` selects subsystems, ports, connectors, and information flows from a complete XMI model and produces a reduced XMI text file.
2. `test_port_information.py` parses the reduced text to recover the source system, output port, conveyed information, target system, and receiving port.

The parser accepts complete XML and the legacy three-part text format labeled `第一部分 / 第二部分 / 第三部分` (Part 1 / Part 2 / Part 3). For the legacy format, it restores namespace declarations and missing Class closing tags without inventing missing connections. Other malformed XML raises an error. When `TRACE_USE_PORT_CONTEXT=false`, the main pipeline skips port-file parsing and uses `is_port=False` for all samples.

## Inputs and Outputs

### Extract XMI Fragments

The input is a complete SysML/XMI file. The output contains only information relevant to ports and information flows. Run from `src/ours`:

```bash
python Port_Information/information_extract.py \
  --xmi path/to/model.xmi \
  --output Data/Port_Information/关于端口信息流动的XMI.txt
```

The default search scope is the following package path:

```text
1 Problem Domain
└── 2 White Box
    └── 2 Conceptual Subsystems Communication
```

Use `--package-path` to select a different scope.

### Build Information-Flow JSON

```python
from Port_Information.test_port_information import parse_port_information

port_information = parse_port_information(
    r"Data\Port_Information\关于端口信息流动的XMI.txt"
)
```

The return value is a JSON string. Each record describes an information flow whose ports can be resolved through a connector. The Chinese filename in the examples matches the pipeline's default port input path.

## Stage 1: XMI Extraction

### 1. Locate the Scope

The extractor follows the configured package names in order. It searches through intermediate container nodes when necessary, but the names and order must match. If the package path cannot be found, extraction fails rather than selecting a different model scope.

### 2. Extract Subsystems and Ports

A subsystem must match:

```xml
<packagedElement xmi:type="uml:Class" ...>
```

The extractor reads the subsystem's `xmi:id` and `name`.

A port must be a direct child of that Class and match:

```xml
<ownedAttribute xmi:type="uml:Port" .../>
```

For each port, the extractor reads:

- Port `xmi:id` and `name`.
- Owning subsystem `xmi:id`.
- Original attributes, including `visibility`, `aggregation`, `isConjugated`, and `type`.

Different subsystems may have ports with identical names. Subsequent matching uses the globally unique `xmi:id`.

### 3. Select Subsystems

Three selection strategies are supported, in the following priority order.

#### Explicit System Names

With `--systems`, match normalized system names exactly:

```bash
--systems VCCU Cooling Heating
```

Normalization trims leading and trailing spaces and collapses consecutive whitespace. It does not perform fuzzy matching.

#### Expand from an Anchor

With `--anchor`, start at the specified system and expand in both directions using the `informationSource` and `informationTarget` of each `InformationFlow`:

```bash
--anchor VCCU --hop 1
```

`hop` is the number of expansion rounds. Each round adds source or target systems connected by information flows to the currently selected systems.

#### Default Selection

Without `--systems` or a valid `--anchor`, retain all Classes containing ports within the selected scope.

### 4. Extract Connectors

A connector must match:

```xml
<ownedConnector xmi:type="uml:Connector" ...>
```

Its endpoints reference port IDs through `role` on `<end>` elements:

```xml
<end role="port_xmi_id"/>
```

A connector is retained when it has at least one valid `role` and every role refers to a port in the selected subsystems. Connectors with an endpoint outside the selected scope are excluded.

### 5. Extract Information Flows

An information flow must match:

```xml
<packagedElement xmi:type="uml:InformationFlow" ...>
```

The extractor reads:

- `name`: information-flow name.
- `informationSource`: source subsystem `xmi:id`.
- `informationTarget`: target subsystem `xmi:id`.
- `realizingConnector`: connector `xmi:id` implementing the flow.

A flow is retained only if it has `realizingConnector` and at least one referenced connector was selected. Flows without a connector mapping cannot be resolved to specific ports and are excluded from subsequent structured parsing.

### 6. Organize the Reduced XMI

The output has three parts:

1. Subsystem and port definitions.
2. Connectors and their port endpoints.
3. Information flows with source systems, target systems, and realizing connectors.

This preserves the reference chain from systems to ports and from information flows through connectors to their ports.

## Stage 2: Port Information-Flow Parsing

`test_port_information.py` parses the reduced XMI text.

### 1. Index Subsystems and Ports

Parsing `uml:Class` produces:

```text
Subsystem xmi:id -> subsystem name
```

Parsing `uml:Port` inside each Class produces:

```text
Port xmi:id -> port name, owning subsystem ID, owning subsystem name
```

### 2. Index Connectors

Each connector's `role` is interpreted as a port ID:

```text
Connector xmi:id -> [port A ID, port B ID]
```

Subsequent processing assumes binary connectors: each valid connector must connect exactly two ports.

### 3. Parse Information Flows

For each flow, read:

```text
flow_name
informationSource
informationTarget
realizingConnector
```

`informationSource` and `informationTarget` define the direction between systems. `realizingConnector` identifies the actual port connection. The `flow for ` prefix is removed from the flow name to obtain the conveyed information. For example, `flow for Ambient Temperature` yields `Ambient Temperature`.

### 4. Determine Output and Receiving Ports

For each referenced connector, retrieve its two ports and determine the direction from their owning subsystems:

1. If port A belongs to `informationSource`, A is the output port and B is the receiving port.
2. Otherwise, if port B belongs to `informationSource`, B is the output port and A is the receiving port.
3. If neither port belongs to the source system, skip the flow because its direction cannot be resolved reliably.

A final record is generated only when the source system, target system, connector, and both ports are resolved.

### 5. Aggregate Port Directions

The parser also computes direction across all information flows:

- The owning system is a source: `out`.
- The owning system is a target: `in`.
- The owning system is both: `inout`.

The computed `port_dir` is not included directly in the final JSON. Output and receiving ports in that JSON are determined separately for each flow.

## Output JSON

Each valid information flow has the following format. Field names remain in Chinese to match the actual parser output.

```json
{
  "信息流名称": "flow for Ambient Temperature",
  "流动信息": "Ambient Temperature",
  "源系统": "VCCU",
  "信息输出端口": "p2",
  "信息输出端口ID": "source_port_xmi_id",
  "目标系统": "Processing",
  "信息接收端口": "p2",
  "信息接收端口ID": "target_port_xmi_id",
  "连接器ID": "connector_xmi_id"
}
```

| Output key | Meaning |
|---|---|
| `信息流名称` | Original information-flow name |
| `流动信息` | Conveyed information |
| `源系统` | Source system |
| `信息输出端口` | Output port name |
| `信息输出端口ID` | Output port ID |
| `目标系统` | Target system |
| `信息接收端口` | Receiving port name |
| `信息接收端口ID` | Receiving port ID |
| `连接器ID` | Connector ID |

These records tell the LLM which subsystem owns a port, whether it sends or receives information, what information it carries, which subsystem and port lie at the other end, and which connector supports that interpretation.

## Complete Resolution Process

1. Locate `uml:Class` by system name.
2. Find the port within the Class and obtain its unique `xmi:id`.
3. Determine flow direction from the source and target systems.
4. Resolve `realizingConnector` to the connector.
5. Resolve both connector `role` references to actual ports.
6. Identify the sending and receiving ends by comparing port ownership with `informationSource`.
7. Combine flow semantics, system direction, and port connections into structured JSON.
8. Supply this context to subsequent port–requirement traceability reasoning.

## Constraints

- `xmi:id` uniquely identifies systems, ports, and connectors. Names are used for display and initial lookup.
- Ports with the same name may exist in different subsystems; matching across subsystems must use IDs.
- An information flow must reference a valid `realizingConnector` to resolve its ports.
- All connector ports must belong to the selected systems.
- The parser assumes two ports per connector.
- Legacy-format compatibility covers the three-part format described above. For new data, use normalized XML from the extractor with complete namespaces and reference IDs.
