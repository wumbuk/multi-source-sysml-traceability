from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import PROJECT_ROOT
# from LLM_test.Data.Group import group_and_save_to_json_A,group_and_save_to_json_B
from enlarge_function.Initial_EnlargeModelElement import generate_description
# from LLM_test.ThinkingLinks.traverse_Json import traverse_json, traverse_ElementPair_grouped



#因为数据集构造的是元素对，所以在提取模型元素时要先对元素进行去重

#=========2️⃣对元素进行去重操作(根据模型元素的所有属性去重)=================
# pd.set_option('display.max_columns', None)

#对模型元素进行去重操作
def drop_duplicates(df,subset):
    """
    :param df:需要去重的数据集合
    :param subset:去重参考的属性列
    :return:去重过后的数据集
    """
    return df.drop_duplicates(subset=subset,keep='first')


#生成模型元素功能性描述的csv文件，并聚合为json文件，便于后续获取模型上下文
def makeup_context_A(input_df):
    df = input_df.copy()
    df["contextA"] = ""

    print("Generating ElementA functional descriptions via LLM...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = "" if pd.isna(row["NameA"]) else str(row["NameA"])
        path = "" if pd.isna(row["pathA"]) else str(row["pathA"])
        prop = "" if pd.isna(row["propertyA"]) else str(row["propertyA"])
        element_type = "" if pd.isna(row["typeA"]) else str(row["typeA"])
        try:
            desc = generate_description(name, path, prop, element_type)
        except Exception as e:
            desc = f"ERROR: {str(e)}"

        df.at[idx, "contextA"] = desc

    return df  #返回产生描述后的数据集

def makeup_context_B(input_df):
    df = input_df.copy()
    df["contextB"] = ""

    print("Generating ElementB functional descriptions via LLM...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        name = "" if pd.isna(row["NameB"]) else str(row["NameB"])
        path = "" if pd.isna(row["pathB"]) else str(row["pathB"])
        prop = "" if pd.isna(row["propertyB"]) else str(row["propertyB"])
        element_type = "" if pd.isna(row["typeB"]) else str(row["typeB"])
        try:
            desc = generate_description(name, path, prop, element_type)
        except Exception as e:
            desc = f"ERROR: {str(e)}"

        df.at[idx, "contextB"] = desc

    return df  #返回产生描述后的数据集

def enlarge_element(
    mr_file: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """为模型元素生成 contextA，并返回生成后的 CSV 路径。"""
    input_path = Path(mr_file)
    if not input_path.exists():
        input_path = (
            PROJECT_ROOT / "Data" / "GenerateContext" / "Time_Analysis"
            / input_path
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_path}")

    if output_path is None:
        output_csv = (
            PROJECT_ROOT / "Data" / "GenerateContext" / input_path.name
        )
    else:
        output_csv = Path(output_path)

    MR_df = pd.read_csv(input_path)
    # 支持重复运行：旧 contextA 会由本次语义丰富结果重新生成。
    MR_df = MR_df.drop(columns=["contextA"], errors="ignore")

    # ===== 1️⃣ 提取唯一元素 =====
    mr_model = MR_df[["NameA", "pathA", "propertyA", "typeA"]]
    mr_model = drop_duplicates(
        df=mr_model,
        subset=["NameA","pathA","propertyA","typeA"]
    )

    # ===== 2️⃣ 生成唯一 context =====
    unique_A_with_context = makeup_context_A(mr_model)

    # ===== 3️⃣ 回填到原始数据 =====
    final_df = MR_df.merge(
        unique_A_with_context,
        on=["NameA","pathA","propertyA","typeA"],
        how="left"
    )
    # =====================================================
    # ===== 4️⃣ ⭐ 插入 contextA 到 pathA 后面 ============
    # =====================================================
    cols = list(final_df.columns)

    cols.remove("contextA")  # 先移除
    idx = cols.index("pathA")  # 找到 pathA 位置
    cols.insert(idx + 1, "contextA")  # 插入到后面

    final_df = final_df[cols]

    # ===== 5️⃣ 保存 =====
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"语义丰富结果已保存: {output_csv}")
    print(f"输出字段: {list(final_df.columns)}")
    return output_csv




    #对于MM，使用A，B来存储模型元素，因此需要对A，B均去重并分别保留A，B属性元素为CSV文件
    # MM_ModelA=MM_df.iloc[:,[0,1,2,3]]
    # MM_ModelB=MM_df.iloc[:,[4,5,6,7]]
    # print(MM_ModelA)
    # print(MM_ModelB)
    # MM_ModelA=drop_duplicates(df=MM_ModelA,subset=["NameA","contextA","propertyA","typeA"])
    # MM_ModelB=drop_duplicates(df=MM_ModelB,subset=["NameB","contextB","propertyB","typeB"])
    # print(MM_ModelA)
    # print(MM_ModelB)

    #==========3初步生成元素描述并按照路径聚合为json格式文件，用于后续生成相应模块功能描述，并迭代生成模型元素功能描述====================
    #设置输出csv文件路径,合并csv文件和模型元素文件名称组成路径,根据MR和MM来修改文件路径
    # MM_save_A=r"Model_A.csv"
    # MM_save_B=r"Model_B.csv"
    # Output_csv_A=os.path.join(MF_CSV,MM_save_A)
    # Output_csv_B=os.path.join(MF_CSV,MM_save_B)
    # #生成元素描述,并输出到csv文件
    # makeup_context_A(MM_ModelA,Output_csv_A)
    # makeup_context_B(MM_ModelB,Output_csv_B)



if __name__ == "__main__":
    enlarge_element("Air_Condition.csv")
