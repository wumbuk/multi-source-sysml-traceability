# Dataset Notes

The experiment CSV schema is:

```text
NameA,pathA,contextA,propertyA,typeA,
NameB,pathB,contextB,propertyB,typeB,Relation
```

- Side A normally represents a SysML model element.
- Side B normally represents a requirement element.
- `contextA` and `contextB` contain semantic descriptions.
- `pathA` and `pathB` encode hierarchical model paths.
- `Relation=0` denotes no traceability relation.
- Positive labels use forms such as `1/Refine` and `1/Satisfy`.

The default Air Condition experiment uses:

- `Data/GenerateContext/Air_Condition/Air_Condition.csv`: complete enriched
  dataset.
- `Data/GenerateContext/Air_Condition/KG.csv`: graph/training portion.
- `Data/GenerateContext/Air_Condition/Judge.csv`: evaluation portion.

`Data/Data_Divide.py` performs the project-specific split. Keep the split files,
the split seed, and the exact gold-label revision fixed when comparing methods.

Before publishing this repository, confirm that every included dataset and XMI
fragment may be redistributed under the intended repository license.
