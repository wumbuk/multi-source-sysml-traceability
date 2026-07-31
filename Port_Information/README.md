# Port Information

该组件用于从 SysML/XMI 模型中提取端口及其信息流，并将原始 XMI 关系转换为便于后续追溯关系推理使用的结构化 JSON。

整体处理分为两个阶段：

1. `information_extract.py`：从完整 XMI 中筛选子系统、端口、连接器和信息流，生成精简的 XMI 文本。
2. `test_port_information.py`：解析精简文本，恢复“源系统、输出端口、流动信息、目标系统、接收端口”之间的对应关系。

## 输入与输出

### XMI 片段提取

输入是完整的 SysML/XMI 文件，输出是只包含端口信息流相关内容的文本文件。

```bash
python Port_Information/information_extract.py \
  --xmi path/to/model.xmi \
  --output Data/Port_Information/关于端口信息流动的XMI.txt
```

默认仅在以下 Package 路径中查找：

```text
1 Problem Domain
└── 2 White Box
    └── 2 Conceptual Subsystems Communication
```

可以通过 `--package-path` 修改该范围。

### 信息流 JSON 构建

```python
from Port_Information.test_port_information import parse_port_information

port_information = parse_port_information(
    r"Data\Port_Information\关于端口信息流动的XMI.txt"
)
```

返回值是 JSON 字符串，每条记录表示一条可以由 Connector 定位到具体端口的信息流。

## 第一阶段：XMI 信息提取规则

### 1. 确定解析范围

程序首先按照 Package 名称逐层查找目标作用域。若中间存在额外容器节点，会继续向下搜索，但 Package 名称及顺序必须与配置一致。找不到目标路径时直接报错，避免在错误的模型范围内提取数据。

### 2. 提取子系统及端口

子系统必须满足：

```xml
<packagedElement xmi:type="uml:Class" ...>
```

程序读取：

- 子系统的 `xmi:id`
- 子系统的 `name`

端口必须是该 Class 的直接子元素，并满足：

```xml
<ownedAttribute xmi:type="uml:Port" .../>
```

程序读取：

- 端口的 `xmi:id`
- 端口的 `name`
- 端口所属子系统的 `xmi:id`
- 原始端口属性，例如 `visibility`、`aggregation`、`isConjugated` 和 `type`

端口名称可能在不同子系统中重复，因此后续匹配使用全局唯一的 `xmi:id`，不能只依赖端口名称。

### 3. 选择需要保留的子系统

程序支持三种选择策略，优先级从高到低如下。

#### 指定系统名称

使用 `--systems` 时，按照规范化后的系统名称精确匹配：

```bash
--systems VCCU Cooling Heating
```

名称规范化只会去除首尾空格并合并连续空白，不进行模糊匹配。

#### 按锚点扩展

使用 `--anchor` 时，以指定系统为起点，根据 `InformationFlow` 的 `informationSource` 和 `informationTarget` 进行双向邻接扩展：

```bash
--anchor VCCU --hop 1
```

`hop` 表示扩展轮数。每轮都会加入与当前已选系统存在信息流连接的源系统或目标系统。

#### 默认选择

如果没有提供 `--systems` 或有效的 `--anchor`，默认保留作用域内所有包含端口的 Class。

### 4. 提取 Connector

Connector 必须满足：

```xml
<ownedConnector xmi:type="uml:Connector" ...>
```

其端点通过 `<end>` 元素中的 `role` 指向端口 `xmi:id`：

```xml
<end role="port_xmi_id"/>
```

Connector 的保留规则是：

- Connector 至少包含一个有效的 `role`；
- 所有 `role` 都必须属于当前已选子系统的端口集合。

因此，只连接了部分目标端口、另一端位于选择范围之外的 Connector 不会进入输出。

### 5. 提取 InformationFlow

信息流必须满足：

```xml
<packagedElement xmi:type="uml:InformationFlow" ...>
```

程序读取：

- `name`：信息流名称；
- `informationSource`：源子系统 `xmi:id`；
- `informationTarget`：目标子系统 `xmi:id`；
- `realizingConnector`：实现该信息流的 Connector `xmi:id`。

信息流只有同时满足以下条件才会被保留：

- 包含 `realizingConnector`；
- 至少一个 `realizingConnector` 引用了已选 Connector。

没有 Connector 映射的信息流无法确定具体经过哪两个端口，因此不会进入后续结构化解析。

### 6. 精简 XMI 的组织方式

输出文本分为三部分：

1. 子系统及其端口定义；
2. Connector 及其端口端点；
3. InformationFlow 及其源系统、目标系统和实现 Connector。

该结构保留了从“系统”到“端口”，再从“信息流”经由“Connector”定位到端口的完整引用链。

## 第二阶段：端口信息流解析规则

`test_port_information.py` 对精简 XMI 文本进行逐行解析。

### 1. 建立子系统和端口索引

解析 `uml:Class` 得到：

```text
子系统 xmi:id -> 子系统名称
```

解析 Class 内的 `uml:Port` 得到：

```text
端口 xmi:id -> 端口名称、所属子系统 ID、所属子系统名称
```

### 2. 建立 Connector 索引

每个 Connector 下的 `role` 被解释为端口 ID：

```text
Connector xmi:id -> [端口 A ID, 端口 B ID]
```

当前后续逻辑按二元 Connector 处理，即每个有效 Connector 应恰好连接两个端口。

### 3. 解析 InformationFlow

每条信息流读取：

```text
flow_name
informationSource
informationTarget
realizingConnector
```

其中：

- `informationSource` 和 `informationTarget` 表示系统级方向；
- `realizingConnector` 表示该信息流实际经过的端口连接；
- 信息流名称中的 `flow for ` 前缀会被移除，剩余内容作为“流动信息”。

例如：

```text
flow for Ambient Temperature
```

会得到：

```text
Ambient Temperature
```

### 4. 确定输出端口和接收端口

对于 InformationFlow 引用的 Connector，程序取得其两个端口，并根据端口所属子系统判断方向：

1. 若端口 A 所属子系统等于 `informationSource`，端口 A 是输出端口，端口 B 是接收端口；
2. 否则，若端口 B 所属子系统等于 `informationSource`，端口 B 是输出端口，端口 A 是接收端口；
3. 若两个端口都不属于源系统，则无法建立可靠方向，该信息流被跳过。

只有源系统、目标系统、Connector 和两个端口都能成功解析时，才生成最终记录。

### 5. 端口方向统计

程序还会根据所有信息流统计端口方向：

- 端口所属系统是信息源：`out`
- 端口所属系统是信息目标：`in`
- 同时作为信息源和信息目标：`inout`

当前 `port_dir` 已完成计算，但没有直接写入最终 JSON。最终 JSON 中的输出端口和接收端口是针对每一条具体信息流动态确定的。

## 最终 JSON 结构

每条有效信息流的输出格式如下：

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

该结果向后续 LLM 提供以下证据：

- 当前端口属于哪个子系统；
- 端口是信息发送端还是接收端；
- 端口传递的具体信息；
- 信息流连接的目标子系统及另一端端口；
- 结论对应的 Connector 标识。

## 完整追溯过程

端口信息流的恢复过程可以概括为：

1. 根据系统名称找到 `uml:Class`；
2. 在 Class 内根据端口名称找到端口，并取得唯一 `xmi:id`；
3. 根据 `InformationFlow` 的源系统和目标系统确定信息流方向；
4. 根据 `realizingConnector` 找到实现该信息流的 Connector；
5. 根据 Connector 两端的 `role` 找到实际连接的两个端口；
6. 根据端口所属系统与 `informationSource` 的对应关系确定发送端和接收端；
7. 将信息流语义、系统方向和端口连接关系组合为结构化 JSON；
8. 将结构化端口上下文提供给后续端口与需求追溯关系推理。

## 关键约束

- `xmi:id` 是系统、端口和 Connector 的唯一标识，名称仅用于展示和初步定位。
- 不同子系统可以存在同名端口，不能跨系统仅按端口名称匹配。
- InformationFlow 必须引用有效的 `realizingConnector`，否则无法定位具体端口。
- Connector 的端口必须都位于已选择的系统范围内。
- 当前解析器假设一个 Connector 对应两个端口。
- 当前文本解析依赖提取器生成的规范化、逐行 XMI 片段，不建议直接将任意格式的完整 XMI 交给 `test_port_information.py`。
