import pandas as pd
from datasets import Dataset 
from strong_reject.evaluate import evaluate_dataset

# 1. 🚨 配置您的文件路径和评估器
# 请将 'my_responses.csv' 替换为您的实际 CSV 文件路径
CSV_FILE_PATH = "/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/strong_reject/strong_reject/strong_reject/prompt_and_output.csv" 
EVALUATOR = "strongreject_finetuned"

# --- 步骤 1: 从 CSV 文件加载数据并确保列名正确 ---
try:
    # 使用 pandas 读取 CSV 文件
    responses_df = pd.read_csv(CSV_FILE_PATH)
    
    # --- 关键：确保数据集包含必需的 'forbidden_prompt' 列 ---
    # 假设您的输入提示列名是 'prompt'，则将其重命名为 'forbidden_prompt'
    # 如果您的提示列名已经是 'forbidden_prompt'，则不需要重命名。
    if 'prompt' in responses_df.columns:
        responses_df = responses_df.rename(columns={'prompt': 'forbidden_prompt'})
        print("💡 提示：已将 'prompt' 列重命名为 'forbidden_prompt'")
    
    # 检查必需的列
    if 'forbidden_prompt' not in responses_df.columns:
        raise KeyError("CSV文件中缺少必需的列 'forbidden_prompt'。请检查您的原始提示列名是否正确。")
        
    if 'response' not in responses_df.columns:
        raise KeyError("CSV文件中缺少必需的列 'response'。")

    # 可选: 添加一个分组ID列，方便最后计算整体平均分
    responses_df['group_id'] = 'my_csv_data'

    print(f"✅ 成功加载 CSV 文件: {CSV_FILE_PATH}")
    print(f"共加载 {len(responses_df)} 条记录。")
    print("-" * 40)
    
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {CSV_FILE_PATH}。请检查路径是否正确。")
    exit()
except KeyError as e:
    print(f"❌ 错误: 列名错误，{e}")
    exit()
except Exception as e:
    print(f"❌ 错误: 读取 CSV 文件时发生异常: {e}")
    exit()


# --- 步骤 2: 转换为 StrongREJECT 兼容的 Dataset 对象 ---
responses_dataset = Dataset.from_pandas(responses_df) 


# --- 步骤 3: 使用 StrongREJECT 评估器评分 ---
# 注意：此步骤依赖于您已完成 Mac 上的 offload_folder 源码修改
print(f"🚀 正在使用评估器 '{EVALUATOR}' 进行有害性评分...")
eval_dataset = evaluate_dataset(responses_dataset, [EVALUATOR])
print("✅ 评估完成。")
print("-" * 40)

# --- 步骤 4: 结果分析 ---
results_df = eval_dataset.to_pandas()

# 计算整体平均得分
average_scores = results_df.groupby('group_id')["score"].mean()

print("\n📊 最终评估结果 (平均有害性得分):")
print(f"您的数据集 '{CSV_FILE_PATH}' 的平均得分: {average_scores.iloc[0]:.4f}")

# 预览带有详细得分的表格
print("\n📝 详细得分预览 (得分越高越有害):")
print(results_df[['forbidden_prompt', 'response', 'score']].head(5))