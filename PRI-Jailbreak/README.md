# Progressive Role Immersion Jailbreak (PRI-Jailbreak)

本开源项目实现了论文中的**“融合渐进式角色沉浸的多轮越狱攻击框架” (Multi-turn Jailbreak Attack Framework Integrating Progressive Role Immersion)**。

## 项目简介

该框架旨在通过多轮对话，逐渐为大语言模型（LLM）构建一个深度的虚拟角色场景（角色沉浸）。在模型逐渐接受该设定、防御机制降低之后，在随后的对话轮次中引入恶意的载荷（Payload），从而诱导模型输出被对齐机制拒绝的内容。

主要设计特点：
- **一致性角色构建 (Consistent Role Building)**：利用多轮交互平滑引导模型进入角色。
- **渐进式攻击 (Progressive Attack)**：避免“一步到位”引起的安全过滤，分阶段深入。
- **自动化测试评估**：包含用于构建目标、攻击执行与指标评估（越狱成功率）的自动化接口。

## 目录结构

```
PRI-Jailbreak/
├── pri_jailbreak/          # 核心代码包
│   ├── core/               # 攻击逻辑与角色管理模块
│   │   ├── attacker.py     # 多轮攻击调度
│   │   ├── role_manager.py # 渐进式角色构建
│   │   └── target_model.py # 目标大语言模型接口
│   ├── evaluators/         # 越狱判定器（Judge）
│   └── utils/              # 辅助工具（日志等）
├── examples/               # 示例脚本
├── data/                   # 数据集/测试用例存放位置
├── requirements.txt        # 依赖项
└── README.md               # 项目说明
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行越狱测试
参考 `examples/run_attack.py` 作为入口点进行测试。

```bash
python examples/run_attack.py
```
