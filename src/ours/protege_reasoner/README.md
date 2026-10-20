# Java / SWRL Ontology Reasoner

English | [中文](README_zh-CN.md)

`use_reasoner.py` invokes the Java reasoner using settings from `config.py`.

- Ontology: `kg-reasoner/ontology/KG.ttl`.
- Built executable JAR: `kg-reasoner/build/libs/kg-reasoner-1.0-SNAPSHOT.jar`.
- Java source: `kg-reasoner/src/main/java/com/example/SwrlReasoner.java`.

To build the JAR, or rebuild after editing Java source, run `./gradlew jar` from `kg-reasoner` (`.\gradlew.bat jar` on Windows). Use a JDK compatible with Gradle 8.6, such as JDK 17.

To change SWRL rules, edit the ontology in Protégé, export it as TTL, and replace `ontology/KG.ttl`. Alternatively, set `TRACE_ONTOLOGY_PATH` to another ontology file.
