import pandas as pd

def calculate_asr_stats(file_path):
    """
    读取CSV文件，计算不同角色和行为对应的危害度评分总分及平均分，并直接输出到控制台。

    CSV文件要求包含以下列（按顺序）：
    1. 角色 (Role)
    2. 行为 (Action)
    3. 忽略列 (需存在，但不读取内容)
    4. 忽略列 (需存在，但不读取内容)
    5. 危害度评分 (Hazard Score)

    Args:
        file_path (str): 您的 CSV 文件的完整路径。
    """
    try:
        # 1. 读取CSV：只取第0（角色）、1（行为）、4（评分）列
        df = pd.read_csv(
            file_path, 
            header=None, 
            encoding='utf-8',
            usecols=[0, 1, 4], 
            names=['角色', '行为', '危害度评分']
        )
    except Exception as e:
        print(f"错误: 无法处理文件 '{file_path}'。请检查路径、文件编码或格式。详细信息: {e}")
        return

    # 数据清理：确保评分为数值，并移除缺失值
    df['危害度评分'] = pd.to_numeric(df['危害度评分'], errors='coerce')
    df.dropna(subset=['危害度评分'], inplace=True)

    print("\n" + "=" * 30)
    print(f"ASR 统计结果 ({file_path})")
    print("=" * 30)

    # 2. 角色总分统计 (Group by '角色', calculate sum)
    print("\n[一] 各角色的总危害度评分：")
    role_totals = df.groupby('角色')['危害度评分'].sum()
    for role, total in role_totals.items():
        print(f"角色: {role: <10} | 总分: {total:.2f}")

    # 3. 行为总分和平均分统计 (Group by '行为', calculate sum and mean)
    print("\n[二] 各行为的总分与平均危害度评分：")
    action_stats = df.groupby('行为')['危害度评分'].agg(['sum', 'mean'])
    for action, row in action_stats.iterrows():
        print(f"行为: {action: <10} | 总分: {row['sum']:.2f} | 平均分: {row['mean']:.2f}")


if __name__ == '__main__':
    your_file_path = '/Users/panmingyang/pyproject/llmsecurity/jailbreak_secLLM/eval/eval_result.csv'  # <-- 在这里填写您的文件路径
    calculate_asr_stats(your_file_path)