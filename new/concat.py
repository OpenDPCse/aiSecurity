#拼接（role，prompt，question_ctgr,question,concat_prompt_question）
import pandas as pd
import os

# 定义文件名
HACKER_PROMPT_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/my_data/hacker_prompt.csv'
NON_HACKER_PROMPT_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/my_data/non_hacker_prompt.csv'
HACKING_QUESTION_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/my_data/hacking_question.csv'
NON_HACKING_QUESTION_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/my_data/non_hacking_question.csv'
OUTPUT_FILE = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/newMethod/my_data/concat_prompt.csv'

# 假设 prompt 文件中的列名为 'prompt'
# 并且所有文件的列名如您所述
PROMPT_COL = 'prompt' 
QUESTION_COLS = ['question_ctgr', 'question']

def load_data(file_path, columns):
    """加载 CSV 文件并确保列名正确"""
    try:
        df = pd.read_csv(file_path)
        # 确保 DataFrame 只有需要的列
        if PROMPT_COL in columns and PROMPT_COL not in df.columns:
             # 如果是 prompt 文件，假设第一列是 prompt，根据您的描述，列名可能是缺失的，我们进行重命名
             if len(df.columns) == 1:
                df.columns = [PROMPT_COL]
             else:
                # 假设 prompt 列就是第一个列名（如果有多个列）
                df = df[[df.columns[0]]].rename(columns={df.columns[0]: PROMPT_COL})
        
        elif all(col in columns for col in QUESTION_COLS) and not all(col in df.columns for col in QUESTION_COLS):
             # 如果是 question 文件，假设前两列是 question_ctgr 和 question
             if len(df.columns) >= 2:
                 df.columns = QUESTION_COLS + list(df.columns[2:])
                 df = df[QUESTION_COLS]
             else:
                 raise ValueError(f"文件 {file_path} 的列数不足，期望至少有 {len(QUESTION_COLS)} 列。")

        return df[columns]
    except FileNotFoundError:
        print(f"错误：文件未找到 - {file_path}")
        return pd.DataFrame(columns=columns)
    except Exception as e:
        print(f"加载文件 {file_path} 时发生错误: {e}")
        return pd.DataFrame(columns=columns)

def create_cartesian_product(prompt_df, question_df, role):
    """
    执行笛卡尔积 (Cross Join) 合并，并创建新列。
    使用 how='cross' 实现笛卡尔积。
    """
    if prompt_df.empty or question_df.empty:
        return pd.DataFrame()

    # 1. 执行笛卡尔积 (Cross Join)
    # pd.merge(how='cross') 可以生成所有组合
    combined_df = pd.merge(prompt_df, question_df, how='cross')
    
    # 2. 添加 'role' 列
    combined_df.insert(0, 'role', role)

    # 3. 创建 'concat_prompt_question' 列
    # 假设拼接方式是 "prompt + question"
    combined_df['concat_prompt_question'] = combined_df[PROMPT_COL].astype(str) + " " + combined_df['question'].astype(str)
    
    # 4. 调整最终列的顺序
    final_columns = ['role', PROMPT_COL, 'question_ctgr', 'question', 'concat_prompt_question']
    
    return combined_df[final_columns]

# --- 主执行部分 ---
print("开始加载数据...")

# 1. 加载 Prompt 数据
# 假设 hacker_prompt, non_hacker_prmopt 的列名都是 'prompt'
hacker_prompts = load_data(HACKER_PROMPT_FILE, [PROMPT_COL])
non_hacker_prompts = load_data(NON_HACKER_PROMPT_FILE, [PROMPT_COL])

# 2. 加载 Question 数据
# 假设 hacking_question, non_hacking_question 的列名都是 ('question_ctgr', 'question')
all_questions = pd.concat([
    load_data(HACKING_QUESTION_FILE, QUESTION_COLS),
    load_data(NON_HACKING_QUESTION_FILE, QUESTION_COLS)
], ignore_index=True)

if hacker_prompts.empty and non_hacker_prompts.empty:
    print("错误：没有加载到任何 Prompt 数据。")
elif all_questions.empty:
    print("错误：没有加载到任何 Question 数据。")
else:
    print(f"已加载 {len(hacker_prompts)} 条 Hacker Prompt, {len(non_hacker_prompts)} 条 Non-Hacker Prompt。")
    print(f"已加载 {len(all_questions)} 条 Question。")
    print("开始生成笛卡尔积组合...")

    # 3. 生成组合 A: Hacker Prompt + 所有 Question
    df_hacker = create_cartesian_product(hacker_prompts, all_questions, role='hacker')

    # 4. 生成组合 B: Non-Hacker Prompt + 所有 Question
    df_non_hacker = create_cartesian_product(non_hacker_prompts, all_questions, role='non_hacker')

    # 5. 合并最终结果
    final_df = pd.concat([df_hacker, df_non_hacker], ignore_index=True)

    # 6. 保存到新的 CSV 文件
    if not final_df.empty:
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✨ 成功生成 {len(final_df)} 条记录并保存到文件: **{OUTPUT_FILE}**")
        print("\n新文件的列结构为: (role, prompt, question_ctgr, question, concat_prompt_question)")
    else:
        print("未生成任何有效组合，文件未创建。")