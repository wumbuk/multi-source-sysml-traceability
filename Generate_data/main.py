import pandas as pd
import os

# 读取原始数据集，这里假设您要处理的是 Data/GenerateContext 下的数据集
# 请根据实际需要修改文件路径
input_file = r"../Data/Model_Requirement_Dataset/Air_Condition.csv"
# 读取CSV文件
df = pd.read_csv(input_file)

# 按照每三个取一个作为测试集 (通过索引取余数，例如索引为 0, 3, 6... 的作为测试集)
# 其余的作为训练集
test_mask = df.index % 3 == 0

# 划分数据集
df_judge = df[test_mask]           # 也就是您需要的 Judge 数据集 (测试集)
df_train = df[~test_mask]          # 也就是您需要的 Model_Requirement 数据集 (训练集)

# 将划分好的数据集分别保存
train_output = r"../Data/GenerateContext/Model_Requirement.csv"
test_output = r"../Data/GenerateContext/Judge.csv"

df_train.to_csv(train_output, index=False, encoding='utf-8')
df_judge.to_csv(test_output, index=False, encoding='utf-8')

print(f"数据划分完成！")
print(f"训练集 ({train_output}) 包含了 {len(df_train)} 条数据。")
print(f"测试集 ({test_output}) 包含了 {len(df_judge)} 条数据。")