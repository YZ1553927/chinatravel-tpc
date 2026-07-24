# TPC Agent: Neuro-Symbolic Travel Planning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于神经符号 AI 的中国城市旅游自动规划 Agent。输入自然语言需求，生成逐日行程计划（含交通、酒店、景点、餐厅、费用明细）。

## 我的贡献

本仓库的核心实现位于 `tpc_agent/` 目录（**5,437 行原创代码**）：

```
tpc_agent/
├── agent.py        (3,651行)  DFS搜索引擎 + 符号约束验证 + 回溯规划
├── llm.py          (1,116行)  自然语言→符号约束的LLM翻译
├── converter.py    (547行)    自然语言→结构化JSON解析器
└── utils.py        (123行)    时间/费用计算 + MMR多样性算法
```

基于南京大学 LAMDA-NeSy 组的 [ChinaTravel](https://github.com/LAMDA-NeSy/ChinaTravel) 框架（见 [ATTRIBUTION.md](ATTRIBUTION.md)）。

### 技术创新点

- **DFS + 符号回溯搜索**：在城际交通、酒店、景点、餐厅、市内交通五层搜索空间中做深度优先搜索，约束失败时回溯
- **NL2SL 翻译**：将自然语言查询转化为可执行的 Python 约束代码（35+ 语义函数）
- **MMR 多样性优化**：使用 Maximal Marginal Relevance 算法在景点/餐厅选择中平衡相关性和多样性
- **统一约束验证**：7 项常识约束 + 用户硬约束 + 偏好优化三层验证体系

## 快速开始

### 安装

```bash
git clone https://github.com/YZ1553927/chinatravel-tpc.git
cd chinatravel-tpc
pip install -e .
```

### 使用示例

```python
from tpc_agent import TPCAgent
from chinatravel.agent.load_model import init_llm

# 创建 LLM 后端
llm = init_llm("deepseek")  # 或 "gpt-4o", "Qwen3-4B"

# 创建 Agent
agent = TPCAgent(
    env=world_env,           # 环境模拟器
    backbone_llm=llm,
    cache_dir="cache/",
)

# 自然语言查询
result = agent.run({
    "nature_language": "一家三口从南京去北京玩两天，预算3000，想看故宫和长城",
    "start_city": "南京",
    "target_city": "北京",
    "days": 2,
    "people_number": 3,
})

print(result["plan"])  # 逐日行程
```

### 输出示例

```json
[{
  "day": 1,
  "activities": [
    {"type": "intercity_transport", "position": "G102", "start_time": "07:00", "end_time": "11:30", "cost": 1450},
    {"type": "attraction", "position": "故宫博物院", "start_time": "13:00", "end_time": "17:00", "cost": 180},
    {"type": "restaurant", "position": "全聚德烤鸭店", "start_time": "18:00", "end_time": "19:30", "cost": 350},
    {"type": "accommodation", "position": "如家快捷酒店", "room_type": "标准间", "cost": 280}
  ]
}]
```

## 架构

```
用户查询 (NL)
    │
    ▼
┌─────────────────────────┐
│  tpc_agent/llm.py        │  ← NL → 符号约束翻译
│  tpc_agent/converter.py  │  ← NL → 结构化JSON
└──────────┬──────────────┘
           │ 符号约束 (Python代码)
           ▼
┌─────────────────────────┐
│  tpc_agent/agent.py      │  ← DFS搜索 + 约束验证
│  ┌── 搜索城际交通         │
│  ├── 搜索酒店             │
│  ├── 搜索景点 (MMR排序)    │
│  ├── 搜索餐厅 (MMR排序)    │
│  ├── 搜索市内交通 (Dijkstra)│
│  └── 约束验证 → 回溯      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  chinatravel/             │  ← 南大框架 (环境+验证)
│  ├── environment/tools    │  数据库API
│  ├── symbol_verification  │  符号验证引擎
│  └── agent/nesy_*         │  NL2SL桥接
└──────────────────────────┘
           │
           ▼
      逐日行程 JSON
```

## 仓库结构

```
├── tpc_agent/              ★ 我的代码 (5,437行)
│   ├── __init__.py         公开API导出
│   ├── agent.py            TPCAgent: DFS搜索 + 约束验证 + 回溯
│   ├── llm.py              TPCLLM: 自然语言 → 符号约束翻译
│   ├── converter.py        SimpleTravelConverter: NL → 结构化JSON
│   └── utils.py            时间/费用计算、MMR多样性算法
│
├── chinatravel/            南大ChinaTravel框架（依赖）
│   ├── agent/
│   │   ├── base.py         Agent抽象基类
│   │   ├── llms.py         DeepSeek/GPT-4o/GLM/Qwen等LLM后端
│   │   ├── load_model.py   LLM/Agent工厂函数
│   │   ├── utils.py        日志与编码工具
│   │   ├── nesy_agent/     NL2SL翻译 + AST语法检查 + Prompt模板
│   │   └── nesy_verifier/  常识约束 + 个性化约束验证
│   ├── environment/tools/  数据API层
│   │   ├── accommodations/ 酒店查询
│   │   ├── attractions/    景点查询
│   │   ├── restaurants/    餐厅查询
│   │   ├── intercity_transport/ 城际交通（火车/飞机）
│   │   ├── transportation/ 市内交通（地铁/打车/步行）
│   │   └── poi/            POI坐标查询
│   ├── symbol_verification/ 符号验证引擎
│   │   ├── concept_func.py 35+语义约束函数
│   │   ├── commonsense_constraint.py 常识约束检查
│   │   ├── hard_constraint.py        用户硬约束检查
│   │   └── preference.py            偏好优化评估
│   ├── evaluation/         评测工具
│   └── data/               数据集加载
│
├── examples/
│   └── demo.py             运行示例
├── database/ → chinatravel/database  数据文件（软链接）
├── README.md
├── AUTHORS.md              作者声明
├── ATTRIBUTION.md          南大框架归属说明
└── pyproject.toml          Python打包配置
```

## 支持

- **10 个城市**：北京、上海、南京、苏州、杭州、深圳、成都、武汉、广州、重庆
- **4 种 LLM**：DeepSeek / GPT-4o / GLM-4 Plus / 本地 Qwen/Mistral/Llama
- **7 项常识约束**：交通正确性、景点开放时间、餐厅用餐时段、住宿必要性、时间顺序、空间一致性、市内交通有效性

## 依赖

- Python 3.9+
- pandas, numpy, geopy
- transformers, torch (本地LLM)
- openai, tiktoken (API LLM)
- jsonschema, json_repair

## 致谢

感谢南京大学 LAMDA-NeSy 研究组提供的 ChinaTravel 框架和数据集。

## License

我的代码 (tpc_agent/) — MIT License
南大框架 (chinatravel/) — 见原始项目许可
