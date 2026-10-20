# Semantic Enrichment

English | [中文](README_zh-CN.md)

`Get_Model_Function.py` handles repeated model elements. Pass the dataset to `enlarge_element`, which deduplicates model elements and calls `generate_description` to produce their descriptions.

The generated descriptions are inserted into the `contextA` column. The enriched dataset is written to the configured output path; the original workflow uses a `GenerateContext` directory.
