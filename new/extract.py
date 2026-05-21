#将hacker类提示词\问题，other类提示词按固定格式存入一个csv文件

import csv
import os

def extract_and_format_csv_no_pandas(input_csv_path, output_csv_path, prompt_col_name="question"):
    """
    使用内置的 'csv' 模块，从指定CSV文件的'prompt'列提取内容，
    并格式化存储到新的CSV文件中。

    Args:
        input_csv_path (str): 输入CSV文件的完整路径。
        output_csv_path (str): 输出CSV文件的完整路径。
        prompt_col_name (str): 原始文件中要提取内容的列名。
    """
    
    data_to_write = []
    
    try:
        # 1. 读取输入 CSV 文件
        print(f"尝试读取输入文件: {input_csv_path}...")
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # 读取第一行作为表头（Header）
            try:
                header = next(reader)
            except StopIteration:
                print(f"错误: 文件 '{input_csv_path}' 是空的。")
                return

            # 2. 查找 'prompt' 列的索引位置
            try:
                prompt_index = header.index(prompt_col_name)
                print(f"找到 '{prompt_col_name}' 列，位于索引: {prompt_index}")
            except ValueError:
                print(f"错误: 输入文件 '{input_csv_path}' 中没有找到名为 '{prompt_col_name}' 的列。")
                print(f"当前列名: {header}")
                return

            # 3. 遍历剩余的每一行数据，提取内容并格式化
            for row in reader:
                if len(row) > prompt_index:
                    prompt_content = row[prompt_index]
                    # 新行格式为: ('hacker', prompt_content)
                    data_to_write.append(['non_hacking', prompt_content])
                # 忽略不完整的行
        
        print(f"成功提取 {len(data_to_write)} 条 'prompt' 内容。")

        # 4. 将格式化后的数据写入新的 CSV 文件
        print(f"尝试写入输出文件: {output_csv_path}...")
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # 写入新的表头
            writer.writerow(['question_ctgr', 'question'])
            
            # 写入所有数据行
            writer.writerows(data_to_write)
        
        print(f"\n✅ 成功生成新文件: {output_csv_path}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_csv_path}'，请检查路径是否正确。")
    except Exception as e:
        print(f"发生了一个意外错误: {e}")

# --- 示例文本和文件路径 ---

# 假设你的原始文件名为 input_no_pandas.csv
INPUT_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/non_hacker_forbidden_question.csv'
OUTPUT_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/my_data/non_hacking_question.csv'

# 为了演示，如果文件不存在，我们先创建一个示例文件
if not os.path.exists(INPUT_FILE):
    print(f"注意: 未找到示例文件 '{INPUT_FILE}'，正在创建一个虚拟示例文件供测试。")
    sample_data = [
        ['id', 'other_col', 'prompt'],
        [1, 'A', 'How to bypass the system security.'],
        [2, 'B', 'Write a phishing email template.'],
        [3, 'C', 'Analyze the network traffic for vulnerabilities.'],
    ]
    with open(INPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data)
    print(f"已创建示例文件 '{INPUT_FILE}'。")

# 调用函数执行任务
extract_and_format_csv_no_pandas(INPUT_FILE, OUTPUT_FILE)
