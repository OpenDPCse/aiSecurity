#计算指标
import pandas as pd
from typing import Dict, Any

def calculate_conditional_averages(file_path: str) -> Dict[str, Any]:
    """
    读取CSV文件，并根据 'role' 和 'question_catgr' 列的条件
    计算 'score' 列的平均值。

    Args:
        file_path: CSV文件的路径（例如: 'your_data.csv'）。

    Returns:
        一个字典，包含所有计算出的平均分。
    """
    try:
        # 1. 读取CSV文件并确保 'score' 为数值类型 (这是必须的)
        df = pd.read_csv(file_path)
        df['score'] = pd.to_numeric(df['score'], errors='coerce')

        # 使用字典存储结果
        results = {}

        # --- 角色平均分计算 (一行代码实现筛选和平均值) ---
        
        # 'role' 为 'hacker' 的平均分
        results['hacker_asr'] = df[df['role'] == 'hacker']['score'].mean()
        
        # 'role' 为 'non_hacker' 的平均分
        results['non_hacker_asr'] = df[df['role'] == 'non_hacker']['score'].mean()

        # --- 题目类别平均分计算 (一行代码实现筛选和平均值) ---
        
        # 'question_catgr' 为 'hacking' 的平均分
        results['hacking_q_asr'] = df[df['question_ctgr'] == 'hacking']['score'].mean()
        
        # 'question_catgr' 为 'non_hacking' 的平均分
        results['non_hacking_q_asr'] = df[df['question_ctgr'] == 'non_hacking']['score'].mean()

        return results

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 '{file_path}'。请确保文件路径正确。")
        return {}
    except KeyError as e:
        print(f"❌ 错误: CSV文件中缺少必要的列。请检查是否存在 'role', 'score', 和 'question_catgr' 三列。缺失列: {e}")
        return {}
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        return {}

# --- 如何使用 (请将 'path/to/your_file.csv' 替换为你的真实文件路径) ---
if __name__ == '__main__':
    # 替换成您的真实文件路径
    your_file_path = 'last_concat.csv' 
    
    # 运行计算
    average_scores = calculate_conditional_averages(your_file_path)

    # 输出结果
    if average_scores:
        print("\n✅ 条件平均分计算结果:")
        for key, value in average_scores.items():
            # 使用 .get() 检查值是否为 NaN (即没有匹配的行)
            if pd.isna(value):
                 print(f"  **{key}:** 0 (或无匹配数据)")
            else:
                 print(f"  **{key}:** {value:.2f}")