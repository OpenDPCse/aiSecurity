"""
DSPy 提示词优化脚本
============================
任务背景：
  - 数据集：经 Zeek 处理的 CIC Modbus 数据集（模拟变电站环境中 Modbus 攻击流量）
  - 下游任务：随机森林分类器，目标列为 "Label"（攻击类型），评估指标为 Accuracy
  - 初始提示词：prompt.txt（引导 LLM 生成特征工程代码片段）

优化策略：
  - 使用 DSPy MIPROv2 优化器（黑盒提示优化，无需梯度）
  - 多维度奖励评估：代码可执行性 + 特征语义质量 + 分类器准确率提升
  - ReAct 反馈闭环：特征重要性反馈驱动迭代更新

使用方法：
  pip install dspy-ai scikit-learn pandas numpy

  export ANTHROPIC_API_KEY="your_key"   # 或 OPENAI_API_KEY

  python dspy_prompt_optimizer.py \
      --data_path /path/to/modbus.csv \
      --label_col Label \
      --api_key YOUR_KEY \
      --model anthropic/claude-sonnet-4-20250514 \
      --n_trials 10 \
      --output optimized_prompt.txt
"""

import argparse
import ast
import io
import json
import os
import re
import sys
import textwrap
import traceback
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

import numpy as np
import pandas as pd
import dspy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. 数据集列描述（与 prompt.txt 保持一致）
# ─────────────────────────────────────────────
DATASET_DESCRIPTION = """
该数据集是 Zeek 处理后的 CIC Modbus 数据集。
数据集包含模拟变电站环境中各种 Modbus 协议攻击的网络流量捕获。
这些攻击包括侦察、查询泛洪、加载有效载荷、延迟响应、修改长度参数、
虚假数据注入、堆叠 Modbus 帧、暴力写入和基线重放……
"""

COLUMN_DESCRIPTIONS = {
    "ts":                    "时间戳（datetime）",
    "id.orig_h":             "源 IP 地址（string）",
    "id.orig_p":             "源端口（int）",
    "id.resp_h":             "目的 IP 地址（string）",
    "id.resp_p":             "目的端口（int，通常为 502=Modbus）",
    "service":               "应用层协议名称（string，如 modbus）",
    "duration":              "连接持续时间（float，秒）",
    "orig_bytes":            "请求方发送字节数（int）",
    "resp_bytes":            "响应方发送字节数（int）",
    "modbus_functions":      "Modbus 功能码（int）",
    "modbus_transactions":   "单连接内 Modbus 事务数（int）",
}

# ─────────────────────────────────────────────
# 2. 初始提示词（来自 prompt.txt）
# ─────────────────────────────────────────────
INITIAL_PROMPT = """
数据集"df"已加载到内存中，列名为当前可用特征。（列的数据类型描述可能不准确）。
"<数据集描述>（该数据集是zeek处理后的CIC modbus数据集，数据集包含模拟变电站环境中各种Modbus协议攻击的网络流量捕获。
这些攻击包括侦察、查询泛洪、加载有效载荷、延迟响应、修改长度参数、虚假数据注入、堆叠Modbus帧、暴力写入和基线重放……）"
"df"中每列的格式及样例：
ts(time):sample['2023-03-23 02:31:40.9','2023-03-23 02:31:41.1']
id.orig_h(int):sample['185.175.0.2','185.175.0.2']
id.orig_p(int):sample['49222','49236','49246']
id.resp_h(int):sample['185.175.0.5','185.175.0.5']
id.resp_p(int):sample['502','502']
service(string):sample['modbus','modbus']
duration(double):sample['0.000256','0.000195']
orig_bytes(int):sample['12','12']
resp_bytes(int):sample['10','11']
modbus_functions(int):sample['2','1','4','3']
modbus_transactions(int):sample['1','2','3','4']
这段代码由一位致力于改进预测效果的数据科学专家编写。它是向数据集中添加新列的代码片段。
此代码生成对预测"Label"（类别）的下游随机森林分类算法有用的附加列。
新增的列添加了新的语义信息，即它们利用了关于数据集的真实世界知识。它们可以是例如特征组合、转换、聚合，其中新列是现有列的函数。
列的尺度和偏移无关紧要。请确保所有使用的列都存在。请严格遵守上述列的描述，并考虑数据类型和类别含义。
此代码也会删除列，如果这些列可能是冗余的并且会损害下游分类器的预测性能（特征选择）。删除列可能有助于降低过拟合的风险，尤其是在数据集较小的情况下。
分类器将在包含所生成列的数据集上进行训练，并在一个保留集上进行评估。评估指标是准确率（accuracy）。性能最佳的代码将被选中。
新增的列可以在其他代码块中使用，被删除的列将不再可用。
代码示例
新增：
\'\'\'python
#(Feature name and description)
#Usefulness:(Description why this adds useful realworld knowledge to classify "Class" according to dataset description and attributes.)
(Some pandas code to add a new column for each row in df)
\'\'\'end
删除：
\'\'\'python 
# Explanation why the column XX is dropped 
df . drop (columns = [ 'XX' ] , inplace = True) 
\'\'\'end
注意，每个代码块只生成一个可用的列或删除不用的列
每个代码块以\'\'\'python起始，以\'\'\'end结尾
"""


# ─────────────────────────────────────────────
# 3. DSPy Signatures
# ─────────────────────────────────────────────

class FeatureEngineeringSignature(dspy.Signature):
    """
    根据工业控制协议流量数据集的描述，生成一段用于特征工程的 Python 代码片段。
    代码片段应当新增对随机森林分类器有用的特征列，或删除冗余列。
    每个代码块只处理一个列（新增或删除），以 '''python 开头，以 '''end 结尾。
    """
    system_prompt: str = dspy.InputField(desc="特征工程任务的系统提示词，描述数据集、列格式和代码规范")
    dataset_context: str = dspy.InputField(desc="数据集详细描述：列名、样例值、数据类型及攻击场景语义")
    feedback: str = dspy.InputField(desc="上一轮迭代的反馈：特征重要性排名、分类准确率及改进建议")
    feature_code: str = dspy.OutputField(desc="生成的特征工程代码片段列表，每个代码块独立，以 '''python 开头，以 '''end 结尾")


class PromptRefinementSignature(dspy.Signature):
    """
    基于当前提示词、执行结果和评估指标，优化提示词以提升特征工程代码质量。
    参考强化学习奖励信号：代码可执行性、特征语义质量、分类准确率。
    """
    current_prompt: str = dspy.InputField(desc="当前正在使用的提示词")
    execution_result: str = dspy.InputField(desc="代码执行结果：成功/失败、错误信息")
    accuracy_score: str = dspy.InputField(desc="当前特征集在随机森林上的分类准确率")
    feature_importance: str = dspy.InputField(desc="特征重要性排名（JSON格式）")
    improvement_hints: str = dspy.InputField(desc="改进建议：哪些特征类型对 Modbus 攻击检测更有价值")
    refined_prompt: str = dspy.OutputField(desc="优化后的提示词，保留原有结构但改进语义清晰度和特征引导")


# ─────────────────────────────────────────────
# 4. 核心 DSPy 模块
# ─────────────────────────────────────────────

class FeatureEngineeringModule(dspy.Module):
    """生成特征工程代码的核心模块（使用 ChainOfThought 增强推理）"""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FeatureEngineeringSignature)

    def forward(self, system_prompt: str, dataset_context: str, feedback: str = "无历史反馈，首次生成。"):
        return self.generate(
            system_prompt=system_prompt,
            dataset_context=dataset_context,
            feedback=feedback,
        )


class PromptRefinementModule(dspy.Module):
    """基于反馈优化提示词的模块"""

    def __init__(self):
        super().__init__()
        self.refine = dspy.ChainOfThought(PromptRefinementSignature)

    def forward(self, current_prompt, execution_result, accuracy_score, feature_importance, improvement_hints):
        return self.refine(
            current_prompt=current_prompt,
            execution_result=execution_result,
            accuracy_score=str(accuracy_score),
            feature_importance=feature_importance,
            improvement_hints=improvement_hints,
        )


# ─────────────────────────────────────────────
# 5. 评估工具
# ─────────────────────────────────────────────

def extract_code_blocks(text: str) -> list[str]:
    """从生成文本中提取 '''python ... '''end 代码块"""
    pattern = r"'{3}python\s*(.*?)'{3}end"
    blocks = re.findall(pattern, text, re.DOTALL)
    # 兼容 ```python ... ``` 格式
    if not blocks:
        pattern2 = r"```python\s*(.*?)```"
        blocks = re.findall(pattern2, text, re.DOTALL)
    return [b.strip() for b in blocks]


def execute_feature_code(df: pd.DataFrame, code_blocks: list[str]) -> tuple[pd.DataFrame, str, int]:
    """
    在 df 副本上安全执行特征工程代码块。
    返回：(修改后的df, 执行状态描述, 成功执行的块数)
    """
    df_work = df.copy()
    results = []
    success_count = 0

    for i, block in enumerate(code_blocks):
        # 安全检查：禁止危险操作
        forbidden = ["import os", "import sys", "subprocess", "eval(", "exec(", "__import__"]
        if any(f in block for f in forbidden):
            results.append(f"块{i+1}: ❌ 拒绝执行（含危险操作）")
            continue

        buf_out = io.StringIO()
        buf_err = io.StringIO()
        try:
            local_ns = {"df": df_work, "pd": pd, "np": np}
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                exec(block, local_ns)
            df_work = local_ns["df"]
            success_count += 1
            results.append(f"块{i+1}: ✅ 执行成功")
        except Exception as e:
            err_msg = traceback.format_exc(limit=3)
            results.append(f"块{i+1}: ❌ 执行失败 — {type(e).__name__}: {e}\n{err_msg[:300]}")

    status = "\n".join(results)
    return df_work, status, success_count


def evaluate_with_rf(df: pd.DataFrame, label_col: str = "Label") -> tuple[float, dict]:
    """
    在特征集上训练随机森林，返回 (accuracy, feature_importance_dict)
    """
    if label_col not in df.columns:
        return 0.0, {}

    # 准备特征矩阵
    drop_cols = [label_col, "ts", "id.orig_h", "id.resp_h", "service"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # 编码非数值列
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # 标签编码
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))

    if len(np.unique(y)) < 2:
        return 0.0, {}

    # 分层抽样（小数据集友好）
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    importance = dict(sorted(
        zip(feature_cols, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    ))
    return acc, importance


def compute_reward(
    accuracy: float,
    baseline_accuracy: float,
    exec_success_rate: float,
    n_new_features: int,
) -> float:
    """
    多维度奖励函数（参考论文公式 2-3）：
      R = α * score + (1-α) * Conf
    这里简化为：
      R = w1 * acc_improvement + w2 * exec_quality + w3 * feature_richness
    """
    w1, w2, w3 = 0.6, 0.3, 0.1
    acc_improvement = max(0.0, accuracy - baseline_accuracy)
    feature_richness = min(1.0, n_new_features / 5.0)  # 5 个新特征为满分
    reward = w1 * acc_improvement + w2 * exec_success_rate + w3 * feature_richness
    return round(reward, 4)


def get_improvement_hints(feature_importance: dict, top_k: int = 5) -> str:
    """根据特征重要性生成改进建议"""
    if not feature_importance:
        return "暂无特征重要性数据，建议优先生成 Modbus 功能码相关特征。"

    top_features = list(feature_importance.items())[:top_k]
    low_features = [k for k, v in feature_importance.items() if v < 0.01]

    hints = []
    hints.append(f"当前最重要的特征（Top {top_k}）：")
    for name, imp in top_features:
        hints.append(f"  - {name}: {imp:.4f}")

    if low_features:
        hints.append(f"\n重要性极低（<0.01）的特征，建议删除：{', '.join(low_features[:5])}")

    hints.append("\nModbus 攻击检测建议关注：")
    hints.append("  - 功能码异常（非标准 FC、高频写操作）")
    hints.append("  - 事务频率和字节比率（泛洪攻击特征）")
    hints.append("  - 时间间隔特征（延迟响应攻击）")
    hints.append("  - 字节长度异常（修改长度参数攻击）")
    hints.append("  - 源端口多样性（侦察攻击特征）")

    return "\n".join(hints)


# ─────────────────────────────────────────────
# 6. DSPy 评估指标（用于优化器）
# ─────────────────────────────────────────────

class FeatureQualityMetric:
    """
    DSPy 优化器使用的评估指标类。
    评估生成的特征工程代码在随机森林上的表现。
    """

    def __init__(self, df: pd.DataFrame, label_col: str, baseline_acc: float):
        self.df = df
        self.label_col = label_col
        self.baseline_acc = baseline_acc

    def __call__(self, example: dspy.Example, prediction, trace=None) -> float:
        """
        返回 [0, 1] 的质量分数。
        DSPy 优化器要求返回数值分数。
        """
        code_blocks = extract_code_blocks(prediction.feature_code)
        if not code_blocks:
            return 0.0

        df_new, exec_status, success_count = execute_feature_code(self.df, code_blocks)
        exec_rate = success_count / max(len(code_blocks), 1)

        if success_count == 0:
            return exec_rate * 0.1  # 只给可执行性少量分数

        try:
            acc, importance = evaluate_with_rf(df_new, self.label_col)
            n_new = len([c for c in df_new.columns if c not in self.df.columns])
            reward = compute_reward(acc, self.baseline_acc, exec_rate, n_new)
            return reward
        except Exception:
            return exec_rate * 0.2


# ─────────────────────────────────────────────
# 7. 主优化流程
# ─────────────────────────────────────────────

class ModbusPromptOptimizer:
    """
    完整的 DSPy 提示词优化流程（ReAct 反馈闭环）

    流程：
      1. 初始化：加载数据集，计算基线准确率
      2. 第一阶段（提示初始化）：用初始提示词生成首批特征代码
      3. 第二阶段（强化学习优化）：MIPROv2 黑盒提示优化
      4. 第三阶段（反馈闭环）：特征重要性驱动的 ReAct 迭代
      5. 输出最优提示词和特征代码
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "Label",
        model: str = "anthropic/claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        n_trials: int = 10,
        react_rounds: int = 3,
    ):
        self.df = df
        self.label_col = label_col
        self.n_trials = n_trials
        self.react_rounds = react_rounds
        self.best_prompt = INITIAL_PROMPT
        self.best_accuracy = 0.0
        self.optimization_history = []

        # 初始化 DSPy LM
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        lm = dspy.LM(model=model, max_tokens=2048)
        dspy.configure(lm=lm)

        # 初始化模块
        self.feature_module = FeatureEngineeringModule()
        self.refine_module = PromptRefinementModule()

        # 计算基线准确率（原始特征集）
        print(" 计算基线准确率（使用原始特征）...")
        self.baseline_acc, _ = evaluate_with_rf(df, label_col)
        print(f" 基线准确率: {self.baseline_acc:.4f}")

    def _build_dataset_context(self) -> str:
        """构建数据集上下文描述"""
        col_desc = "\n".join([f"  {col}: {desc}" for col, desc in COLUMN_DESCRIPTIONS.items()])
        sample_rows = self.df.head(3).to_dict(orient="records")
        return f"""
{DATASET_DESCRIPTION}

列名及语义描述：
{col_desc}

数据样例（前3行）：
{json.dumps(sample_rows, ensure_ascii=False, default=str, indent=2)}

攻击类型分布（Label列）：
{self.df[self.label_col].value_counts().to_dict() if self.label_col in self.df.columns else "未知"}
"""

    def run_react_iteration(self, current_prompt: str, feedback: str, round_num: int) -> dict:
        """执行一轮 ReAct 迭代（Reason + Action）"""
        print(f"\n{'='*60}")
        print(f" ReAct 第 {round_num} 轮迭代")
        print(f"{'='*60}")

        dataset_context = self._build_dataset_context()

        # Action：生成特征工程代码
        print(" LLM 生成特征代码...")
        try:
            pred = self.feature_module(
                system_prompt=current_prompt,
                dataset_context=dataset_context,
                feedback=feedback,
            )
            feature_code = pred.feature_code
        except Exception as e:
            print(f" 生成失败: {e}")
            return {"success": False, "error": str(e)}

        # 提取并执行代码块
        code_blocks = extract_code_blocks(feature_code)
        print(f" 提取到 {len(code_blocks)} 个代码块")

        if not code_blocks:
            print("️  未提取到有效代码块，尝试直接执行原始输出...")
            code_blocks = [feature_code]

        df_new, exec_status, success_count = execute_feature_code(self.df, code_blocks)
        exec_rate = success_count / max(len(code_blocks), 1)
        print(f"️  执行结果:\n{exec_status}")
        print(f" 成功执行: {success_count}/{len(code_blocks)} 块")

        # Reason：评估结果
        new_cols = [c for c in df_new.columns if c not in self.df.columns]
        print(f" 新增特征列: {new_cols}")

        acc = self.baseline_acc
        importance = {}
        if success_count > 0:
            try:
                acc, importance = evaluate_with_rf(df_new, self.label_col)
                print(f" 分类准确率: {acc:.4f}（基线: {self.baseline_acc:.4f}，提升: {acc - self.baseline_acc:+.4f}）")
            except Exception as e:
                print(f"️  评估失败: {e}")

        reward = compute_reward(acc, self.baseline_acc, exec_rate, len(new_cols))
        print(f" 综合奖励分数: {reward:.4f}")

        # 记录历史
        iteration_result = {
            "round": round_num,
            "prompt_length": len(current_prompt),
            "n_code_blocks": len(code_blocks),
            "exec_success_rate": exec_rate,
            "n_new_features": len(new_cols),
            "new_features": new_cols,
            "accuracy": acc,
            "accuracy_improvement": acc - self.baseline_acc,
            "reward": reward,
            "feature_importance": dict(list(importance.items())[:10]),
            "exec_status": exec_status,
            "feature_code": feature_code,
        }
        self.optimization_history.append(iteration_result)

        # 更新最优结果
        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.best_prompt = current_prompt
            print(f"发现更优结果！准确率提升至 {acc:.4f}")

        return iteration_result

    def optimize_with_mipro(self) -> str:
        """
        使用 DSPy MIPROv2 进行黑盒提示优化。
        构建 trainset，让优化器自动搜索最优提示词。
        """
        print("\n" + "="*60)
        print("启动 MIPROv2 黑盒提示优化")
        print("="*60)

        dataset_context = self._build_dataset_context()

        # 构建训练集（多样化的 feedback 场景）
        feedback_scenarios = [
            "首次生成，无历史反馈。请重点关注 Modbus 功能码和事务特征。",
            "上一轮准确率 0.82，功能码频率特征最重要。建议添加字节比率特征。",
            "检测到延迟响应攻击特征提取不足，建议添加 duration 分位数特征。",
            "虚假数据注入攻击（False Data Injection）需要 resp_bytes 异常检测。",
            "侦察攻击特征：源端口多样性和短连接持续时间。",
        ]

        trainset = [
            dspy.Example(
                system_prompt=INITIAL_PROMPT,
                dataset_context=dataset_context,
                feedback=fb,
                feature_code="",  # 由优化器填充
            ).with_inputs("system_prompt", "dataset_context", "feedback")
            for fb in feedback_scenarios
        ]

        # 初始化评估指标
        metric = FeatureQualityMetric(self.df, self.label_col, self.baseline_acc)

        # 运行 MIPROv2 优化
        try:
            from dspy.teleprompt import MIPROv2
            optimizer = MIPROv2(
                metric=metric,
                num_candidates=self.n_trials,
                init_temperature=1.0,
                verbose=True,
            )
            optimized_module = optimizer.compile(
                self.feature_module,
                trainset=trainset,
                num_trials=self.n_trials,
                minibatch_size=min(3, len(trainset)),
                minibatch_full_eval_steps=max(1, self.n_trials // 3),
            )
            print("✅ MIPROv2 优化完成")

            # 提取优化后的提示词
            optimized_prompt = self._extract_optimized_prompt(optimized_module)
            return optimized_prompt

        except Exception as e:
            print(f"️  MIPROv2 优化失败: {e}")
            print(" 回退到 BootstrapFewShot 优化器...")
            return self._fallback_bootstrap_optimization(trainset, metric)

    def _fallback_bootstrap_optimization(self, trainset, metric) -> str:
        """MIPROv2 失败时的备选方案：BootstrapFewShot"""
        from dspy.teleprompt import BootstrapFewShot
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
        try:
            optimized_module = optimizer.compile(self.feature_module, trainset=trainset)
            return self._extract_optimized_prompt(optimized_module)
        except Exception as e:
            print(f"⚠️  BootstrapFewShot 也失败: {e}")
            return INITIAL_PROMPT

    def _extract_optimized_prompt(self, optimized_module: dspy.Module) -> str:
        """从优化后的模块中提取提示词"""
        try:
            # 尝试获取 ChainOfThought 内部的提示
            prog = optimized_module.generate
            if hasattr(prog, "extended_signature"):
                sig = prog.extended_signature
                # 提取字段描述作为优化提示词摘要
                fields_desc = "\n".join([
                    f"  {name}: {field.desc}"
                    for name, field in sig.fields.items()
                ])
                return f"{INITIAL_PROMPT}\n\n[优化增强]\n优化字段描述：\n{fields_desc}"
        except Exception:
            pass
        return INITIAL_PROMPT

    def run(self) -> dict:
        """
        执行完整的三阶段优化流程：
          1. 提示初始化
          2. MIPROv2 黑盒优化
          3. ReAct 反馈闭环迭代
        """
        print("\n" + "."*20)
        print("Modbus 流量异常检测 — DSPy 提示词优化")
        print("."*20)

        # ── 阶段 1：提示初始化（首次 ReAct 迭代）────────────
        print("\n 阶段 1：提示初始化")
        result1 = self.run_react_iteration(
            current_prompt=INITIAL_PROMPT,
            feedback="首次生成，无历史反馈。请重点关注 Modbus 功能码异常、事务频率和字节比率。",
            round_num=1,
        )

        # ── 阶段 2：MIPROv2 黑盒提示优化 ─────────────────────
        print("\n 阶段 2：MIPROv2 黑盒提示优化")
        optimized_prompt = self.optimize_with_mipro()

        # ── 阶段 3：ReAct 反馈闭环迭代 ───────────────────────
        print(f"\n 阶段 3：ReAct 反馈闭环（{self.react_rounds} 轮）")
        current_prompt = optimized_prompt
        feedback = get_improvement_hints(
            result1.get("feature_importance", {}),
        )

        for round_num in range(2, self.react_rounds + 2):
            result = self.run_react_iteration(
                current_prompt=current_prompt,
                feedback=feedback,
                round_num=round_num,
            )

            if not result.get("success", True):
                continue

            # 基于结果优化下一轮提示词
            try:
                refined = self.refine_module(
                    current_prompt=current_prompt,
                    execution_result=result.get("exec_status", ""),
                    accuracy_score=result.get("accuracy", 0.0),
                    feature_importance=json.dumps(
                        result.get("feature_importance", {}),
                        ensure_ascii=False
                    ),
                    improvement_hints=get_improvement_hints(
                        result.get("feature_importance", {})
                    ),
                )
                current_prompt = refined.refined_prompt
                print(f"✏️  提示词已优化（长度: {len(current_prompt)} 字符）")
            except Exception as e:
                print(f"⚠️  提示词优化失败: {e}")

            feedback = get_improvement_hints(result.get("feature_importance", {}))

        # ── 总结 ─────────────────────────────────────────────
        print("\n" + "="*60)
        print(" 优化总结")
        print("="*60)
        print(f"基线准确率:       {self.baseline_acc:.4f}")
        print(f"最优准确率:       {self.best_accuracy:.4f}")
        print(f"准确率提升:       {self.best_accuracy - self.baseline_acc:+.4f}")
        print(f"优化迭代轮数:     {len(self.optimization_history)}")

        best_round = max(self.optimization_history, key=lambda x: x.get("accuracy", 0), default={})
        if best_round:
            print(f"最优轮次:         第 {best_round['round']} 轮")
            print(f"最优新增特征:     {best_round.get('new_features', [])}")

        return {
            "best_prompt": self.best_prompt,
            "best_accuracy": self.best_accuracy,
            "baseline_accuracy": self.baseline_acc,
            "optimization_history": self.optimization_history,
        }


# ─────────────────────────────────────────────
# 8. CLI 入口
# ─────────────────────────────────────────────

def create_demo_dataframe() -> pd.DataFrame:
    """创建演示用的模拟数据集（无真实数据时使用）"""
    np.random.seed(42)
    n = 500

    attack_types = [
        "Normal", "Reconnaissance", "QueryFlood", "LoadPayload",
        "DelayedResponse", "ModifyLength", "FalseDataInjection",
        "StackedFrames", "BruteForceWrite", "BaselineReplay"
    ]

    labels = np.random.choice(attack_types, n, p=[0.3] + [0.077]*9 + [0.007])

    df = pd.DataFrame({
        "ts": pd.date_range("2023-03-23 02:31:40", periods=n, freq="100ms"),
        "id.orig_h": np.random.choice(["185.175.0.2", "185.175.0.3", "192.168.1.10"], n),
        "id.orig_p": np.random.randint(49000, 65535, n),
        "id.resp_h": "185.175.0.5",
        "id.resp_p": 502,
        "service": "modbus",
        "duration": np.abs(np.random.exponential(0.001, n)),
        "orig_bytes": np.random.choice([12, 6, 8, 10, 14], n),
        "resp_bytes": np.random.choice([10, 11, 6, 8], n),
        "modbus_functions": np.random.choice([1, 2, 3, 4, 5, 6, 15, 16], n),
        "modbus_transactions": np.random.randint(1, 20, n),
        "Label": labels,
    })
    return df


def main():
    parser = argparse.ArgumentParser(
        description="DSPy 提示词优化脚本 — Modbus 流量特征工程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        示例：
          # 使用 Claude API（推荐）
          python dspy_prompt_optimizer.py \\
              --data_path modbus.csv \\
              --label_col Label \\
              --model anthropic/claude-sonnet-4-20250514 \\
              --n_trials 10 \\
              --react_rounds 3 \\
              --output optimized_prompt.txt

          # 使用 OpenAI API
          python dspy_prompt_optimizer.py \\
              --data_path modbus.csv \\
              --model openai/gpt-4o \\
              --api_key sk-xxx

          # 演示模式（无真实数据）
          python dspy_prompt_optimizer.py --demo
        """),
    )
    parser.add_argument("--data_path", type=str, help="CSV 数据集路径")
    parser.add_argument("--label_col", type=str, default="Label", help="目标列名（默认: Label）")
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-20250514",
                        help="DSPy LM 模型名（默认: anthropic/claude-sonnet-4-20250514）")
    parser.add_argument("--api_key", type=str, help="API 密钥（也可设置环境变量）")
    parser.add_argument("--n_trials", type=int, default=10, help="MIPROv2 优化试验次数（默认: 10）")
    parser.add_argument("--react_rounds", type=int, default=3, help="ReAct 反馈闭环轮数（默认: 3）")
    parser.add_argument("--output", type=str, default="optimized_prompt.txt", help="输出文件路径")
    parser.add_argument("--demo", action="store_true", help="演示模式：使用模拟数据集")
    parser.add_argument("--history_output", type=str, default="optimization_history.json",
                        help="优化历史记录输出路径")
    args = parser.parse_args()

    # 加载数据集
    if args.demo:
        print(" 演示模式：生成模拟 Modbus 数据集...")
        df = create_demo_dataframe()
        print(f" 生成数据集：{df.shape[0]} 行 × {df.shape[1]} 列")
    elif args.data_path:
        print(f" 加载数据集: {args.data_path}")
        df = pd.read_csv(args.data_path)
        print(f" 数据集形状: {df.shape}")
    else:
        print(" 请提供 --data_path 或使用 --demo 模式")
        sys.exit(1)

    print(f" 列名: {list(df.columns)}")
    if args.label_col in df.columns:
        print(f"🏷  标签分布:\n{df[args.label_col].value_counts()}")

    # 运行优化
    optimizer = ModbusPromptOptimizer(
        df=df,
        label_col=args.label_col,
        model=args.model,
        api_key=args.api_key,
        n_trials=args.n_trials,
        react_rounds=args.react_rounds,
    )

    results = optimizer.run()

    # 保存最优提示词
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(results["best_prompt"])
    print(f"\n 最优提示词已保存至: {args.output}")

    # 保存优化历史
    # 序列化时过滤不可序列化的对象
    history_serializable = []
    for item in results["optimization_history"]:
        item_copy = {k: v for k, v in item.items() if k != "feature_code"}
        history_serializable.append(item_copy)

    with open(args.history_output, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_accuracy": results["baseline_accuracy"],
            "best_accuracy": results["best_accuracy"],
            "improvement": results["best_accuracy"] - results["baseline_accuracy"],
            "iterations": history_serializable,
        }, f, ensure_ascii=False, indent=2)
    print(f" 优化历史已保存至: {args.history_output}")

    print(f"\n优化完成！最终准确率: {results['best_accuracy']:.4f}")
    return results


if __name__ == "__main__":
    main()
