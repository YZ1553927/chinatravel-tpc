# TPC Agent: Neuro-Symbolic Travel Planning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于神经符号 AI 的中国城市旅游自动规划 Agent。输入自然语言需求，生成逐日行程计划（含交通、酒店、景点、餐厅、费用明细）。

## 我的贡献

本仓库的核心实现位于 `tpc_agent/` 目录（**5,733 行原创代码**）：

```
tpc_agent/
├── agent.py        (3,667行)  DFS搜索引擎 + 符号约束验证 + 回溯规划
├── llm.py          (1,275行)  自然语言 → 符号约束翻译 + 模型后端调度
├── converter.py    (668行)    自然语言 → 结构化 JSON（本地模型 / API 双后端）
├── _config.py      (47行)     部署级配置加载（.env）
└── utils.py        (123行)    时间/费用计算 + MMR多样性算法
```

基于南京大学 LAMDA-NeSy 组的 [ChinaTravel](https://github.com/LAMDA-NeSy/ChinaTravel) 框架（见 [ATTRIBUTION.md](ATTRIBUTION.md)）。

### 技术创新点

- **DFS + 符号回溯搜索**：在城际交通、酒店、景点、餐厅、市内交通五层搜索空间中做深度优先搜索，约束失败时回溯
- **NL2SL 翻译**：将自然语言查询转化为可执行的 Python 约束代码（35+ 语义函数）
- **POI 名称对齐**：用户口语名自动对齐到沙盒真名（"故宫" → "故宫博物院"），避免生成无法满足的约束
- **统一约束验证**：7 项常识约束 + 用户硬约束 + 偏好优化三层验证体系
- **模型后端可插拔**：同一套代码支持本地 transformers 模型与任意 OpenAI 兼容 API（DeepSeek / 智谱 / 通义 / OpenAI）

## 快速开始

### 1. 安装

```bash
git clone https://github.com/YZ1553927/chinatravel-tpc.git
cd chinatravel-tpc
pip install -e .
```

### 2. 准备环境数据

把 ChinaTravel 数据库放到**仓库根目录**的 `database/`（该目录不入库，需自行下载）：

```
chinatravel-tpc/
└── database/
    ├── attractions/           景点 CSV
    ├── accommodations/        酒店 CSV
    ├── restaurants/           餐厅 CSV
    ├── intercity_transport/   城际交通（火车 / 航班）
    ├── transportation/        地铁线路
    └── poi/                   POI 坐标
```

### 3. 配置模型

推荐使用 DeepSeek API（无需本地 GPU）：

```bash
cp .env.example .env
# 编辑 .env，填入 TPC_API_KEY
```

也可以直接用环境变量：

```bash
export TPC_API_KEY=sk-xxxxxxxx
export TPC_API_BASE=https://api.deepseek.com
export TPC_API_MODEL=deepseek-chat
```

支持任何 OpenAI 兼容端点，改后两项即可换厂商：

| 厂商 | `TPC_API_BASE` | `TPC_API_MODEL` |
|---|---|---|
| DeepSeek | `https://api.deepseek.com` | `deepseek-chat` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-plus` |
| 通义千问 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |

不配置 API key 时会自动回退到本地模型，此时需要把模型放到 `chinatravel/local_llm/` 下（如 `Qwen3-4B`）。

### 4. 运行

命令行示例：

```bash
python examples/demo.py
```

或启动 HTTP 服务（带网页表单，可直接用中文提问）：

```bash
python server.py               # http://127.0.0.1:8000
python server.py --port 8080
```

接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 网页表单 |
| `POST` | `/plan` | 提交 `{"query": "..."}`，返回 `job_id` |
| `GET` | `/plan/{job_id}` | 查询结果（`status` 为 `done` / `partial` / `failed`） |
| `GET` | `/health` | 后端与配置状态 |

`status=partial` 表示搜索没找到满足全部约束的方案，返回的是退而求其次的结果，可通过 `satisfied` 字段区分。

## 编程接口

```python
import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("chinatravel"))

from chinatravel.environment.world_env import WorldEnv
from chinatravel.agent.load_model import init_agent, init_llm

env = WorldEnv()
llm = init_llm("TPCLLM")          # 自动选择 API 或本地模型

agent = init_agent({
    "method": "TPCAgent",
    "env": env,
    "backbone_llm": llm,
    "cache_dir": "cache",
    "log_dir": "cache",
    "debug": False,
})

# 第一步：自然语言 → 约束 DSL
nl = "我们一家三口从成都去北京玩5天，预算1万元，住大床房，往返坐飞机，想去故宫，市内只坐地铁"
ext = llm.nl_to_python_direct(
    natural_language=nl, start_city="成都", target_city="北京",
    days=5, people_number=3,
)

# 第二步：符号搜索生成行程
query = {
    "uid": "demo_001",
    "nature_language": nl,
    "start_city": ext["start_city"],
    "target_city": ext["target_city"],
    "days": ext["days"],
    "people_number": ext["people_number"],
    "hard_logic_py": ext["hard_logic_py"],
}
succ, plan = agent.run(query, prob_idx="demo_001", oralce_translation=False)
print(succ, plan["itinerary"])
```

## 架构

```
用户查询 (NL)
    │
    ▼
┌─────────────────────────┐
│  tpc_agent/converter.py  │  ← NL → 结构化 JSON
│  tpc_agent/llm.py        │  ← 结构化 → 可执行约束 DSL
└──────────┬──────────────┘
           │ 符号约束 (Python 代码)
           ▼
┌─────────────────────────┐
│  tpc_agent/agent.py      │  ← DFS搜索 + 约束验证
│  ├── 搜索城际交通         │
│  ├── 搜索酒店             │
│  ├── 搜索景点             │
│  ├── 搜索餐厅             │
│  ├── 搜索市内交通 (Dijkstra)│
│  └── 约束验证 → 回溯      │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  chinatravel/             │  ← 南大框架 (环境+验证)
│  ├── environment/         │  数据库 API + WorldEnv
│  ├── symbol_verification  │  符号验证引擎
│  └── agent/nesy_*         │  NL2SL桥接
└──────────────────────────┘
           │
           ▼
      逐日行程 JSON
```

## 仓库结构

```
├── tpc_agent/              ★ 核心实现 (5,733行)
│   ├── agent.py            TPCAgent: DFS搜索 + 约束验证 + 回溯
│   ├── llm.py              TPCLLM: 自然语言 → 符号约束翻译
│   ├── converter.py        SimpleTravelConverter: NL → 结构化JSON
│   ├── _config.py          配置加载（.env）
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
│   ├── environment/
│   │   ├── world_env.py    沙盒环境封装
│   │   └── tools/          数据API层
│   ├── symbol_verification/ 符号验证引擎
│   ├── evaluation/         评测工具
│   └── data/               数据集加载
│
├── examples/
│   └── demo.py             运行示例
├── server.py               HTTP 服务
├── database/               数据文件（不入库，需自行下载）
├── .env.example            配置模板
├── README.md
├── AUTHORS.md              作者声明
├── ATTRIBUTION.md          南大框架归属说明
└── pyproject.toml          Python打包配置
```

## 支持

- **10 个城市**：北京、上海、南京、苏州、杭州、深圳、成都、武汉、广州、重庆
- **LLM 后端**：DeepSeek / GPT-4o / GLM-4 Plus / 任意 OpenAI 兼容端点 / 本地 Qwen·Mistral·Llama
- **7 项常识约束**：交通正确性、景点开放时间、餐厅用餐时段、住宿必要性、时间顺序、空间一致性、市内交通有效性

## 依赖

- Python 3.9+
- pandas, numpy, geopy
- openai, tiktoken, json_repair, jsonschema
- transformers, torch（仅本地模型时需要）

## 致谢

感谢南京大学 LAMDA-NeSy 研究组提供的 ChinaTravel 框架和数据集。

## License

我的代码 (tpc_agent/) — MIT License
南大框架 (chinatravel/) — 见原始项目许可
