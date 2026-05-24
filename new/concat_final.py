#将gpt最后的打分拼接上
import pandas as pd

def merge_csv_simple(file_a, file_b, output_file):
    """
    将文件B的最后一列拼接到文件A的末尾，并保存为新文件。
    """
    try:
        # 读取文件 A
        df_a = pd.read_csv(file_a)
        # 提取文件 B 的最后一列，并命名为 'New_Column'
        col_b = pd.read_csv(file_b).iloc[:, -1].rename('New_Column')
        
        # 按列拼接并保存
        pd.concat([df_a, col_b], axis=1).to_csv(output_file, index=False)
        
        print(f"✨ 成功：'{file_a}' 和 '{file_b}' 已合并到 '{output_file}'。")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

# --- 🎯 调用示例 ---
# 请替换为你的实际文件路径
FILE_A = 'newMethodOutput.csv'
FILE_B = 'answerOnly_scored.csv'
OUTPUT_FILE = 'last_concat.csv'

# 假设你已经准备好了这两个文件
merge_csv_simple(FILE_A, FILE_B, OUTPUT_FILE)