import pandas as pd
import os

def extract_and_save_columns(input_file, output_file, column_names):
    """
    从指定的 CSV 文件中提取指定的列，并保存为新的 CSV 文件。

    参数:
    input_file (str): 原始 CSV 文件的路径。
    output_file (str): 新生成的 CSV 文件的路径。
    column_names (list): 想要提取的列名（字符串列表）。
    """
    try:
        # 1. 读取 CSV 文件，并只加载指定的列
        # usecols 参数允许你指定要读取的列名，大大提高了效率
        df = pd.read_csv(input_file, usecols=column_names)

        # 2. 将包含所选列的数据保存到新的 CSV 文件
        # index=False 避免将 DataFrame 的索引写入新的 CSV 文件
        df.to_csv(output_file, index=False)

        print(f"✅ 成功提取以下列并保存到：'{output_file}'")
        print(f"   提取的列名: {column_names}")
        print(f"   源文件: {input_file}")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{input_file}'")
    except ValueError as e:
        print(f"❌ 错误：指定的列名中可能存在错误或文件格式不正确。详细信息: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


# --- 👇 请修改以下参数为你自己的设置 👇 ---

# 你的 CSV 文件名
input_csv = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/strong_reject/strong_reject/strong_reject/newMethodOutput.csv' 

# 你希望生成的新 CSV 文件名
output_csv = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/strong_reject/strong_reject/strong_reject/prompt_and_output.csv' 

# 想要提取的**两个**列名（请确保它们与你文件中的列名完全一致，包括大小写）
# 示例：['姓名', '分数']
columns_to_extract = ['concat_prompt_question', 'model_answer'] 

# --- 👆 请修改以上参数 👆 ---

# 运行函数
extract_and_save_columns(input_csv, output_csv, columns_to_extract)