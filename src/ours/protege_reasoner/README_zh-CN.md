# Java / SWRL 本体推理器

中文 | [English](README.md)

`use_reasoner.py` 根据 `config.py` 配置调用 Java 推理器。

- 本体文件：`kg-reasoner/ontology/KG.ttl`。
- 构建生成的可执行 JAR：`kg-reasoner/build/libs/kg-reasoner-1.0-SNAPSHOT.jar`。
- Java 源码：`kg-reasoner/src/main/java/com/example/SwrlReasoner.java`。

首次构建或修改 Java 源码后，在 `kg-reasoner` 目录执行 `./gradlew jar`（Windows 使用 `.\gradlew.bat jar`）。使用兼容 Gradle 8.6 的 JDK，例如 JDK 17。
修改 SWRL 规则时，在 Protégé 中编辑并导出 TTL，替换 `ontology/KG.ttl`；也可以通过 `TRACE_ONTOLOGY_PATH` 指向其他本体文件。
