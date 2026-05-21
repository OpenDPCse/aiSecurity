#提取csv的最后一列
import pandas as pd
def extract_last_column_to_csv(input_csv_path, output_csv_path):
    """
    读取CSV文件，提取其最后一列的元素，并生成新的CSV文件。

    Args:
        input_csv_path (str): 待读取的输入CSV文件路径。
        output_csv_path (str): 待生成的新CSV文件路径。
    """
    try:
        # 1. 读取CSV文件
        df = pd.read_csv(input_csv_path)

        # 2. 提取最后一列
        # df.iloc[:, -1] 选取所有行（:）和最后一列（-1）
        last_column = df.iloc[:, -1]

        # 3. 将提取的列转换为一个新的 DataFrame，以便写入CSV
        # 这一步是为了确保新CSV有一个列头（默认为提取列的原列名）
        # 也可以使用 last_column.to_frame(name='NewColumnName') 来指定新的列名
        new_df = last_column.to_frame()

        # 4. 写入新的CSV文件
        # index=False 表示不将 DataFrame 的索引作为一列写入CSV
        new_df.to_csv(output_csv_path, index=False)

        print(f"✅ 成功提取文件 '{input_csv_path}' 的最后一列。")
        print(f"➡️ 新文件已保存到 '{output_csv_path}'。")

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{input_csv_path}'。请检查路径是否正确。")
    except Exception as e:
        print(f"❌ 发生了一个错误: {e}")

# --- 示例用法 ---
# 假设你的原始文件名为 'input.csv'，新文件名为 'output_last_column.csv'
INPUT_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/newMethodOutput.csv'
OUTPUT_FILE = 'answerOnly.csv'

# 在运行代码之前，请确保你的工作目录下有 'input.csv' 文件，
# 或者将 INPUT_FILE 替换为文件的完整路径。
extract_last_column_to_csv(INPUT_FILE, OUTPUT_FILE)