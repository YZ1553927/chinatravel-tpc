"""
Demo: 用一句话生成逐日行程。

用法:
    # 1) 先配置模型（二选一）
    #    a. API 后端：在仓库根创建 .env，填入 TPC_API_KEY（见 .env.example）
    #    b. 本地模型：把 Qwen3-4B 放到 chinatravel/local_llm/Qwen3-4B/
    #
    # 2) 运行
    python examples/demo.py

依赖：把 ChinaTravel 数据库放到仓库根的 database/ 目录。
"""

import os
import sys

# 仓库根 + chinatravel 都进 sys.path，方便按包名导入
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "chinatravel"))

from chinatravel.environment.world_env import WorldEnv
from chinatravel.agent.load_model import init_agent, init_llm


QUERY = (
    "我们一家三口从成都出发，去北京玩5天。"
    "预算1万元，住大床房，往返坐飞机，想去故宫，市内只坐地铁。"
)


def main():
    # 1. 环境（沙盒数据）
    print("[1/4] 加载沙盒环境...")
    env = WorldEnv()

    # 2. 模型后端：有 API key 走 API，否则回退本地模型
    print("[2/4] 初始化模型后端...")
    llm = init_llm("TPCLLM")
    converter = getattr(llm, "travel_converter", None)
    if converter is None:
        print("  ✗ 没有可用的模型后端。")
        print("    方案一：复制 .env.example 为 .env，填入 TPC_API_KEY")
        print("    方案二：把 Qwen3-4B 放到 chinatravel/local_llm/Qwen3-4B/")
        return
    print(f"  后端 = {converter.name}")

    # 3. Agent
    print("[3/4] 初始化 Agent...")
    agent = init_agent({
        "method": "TPCAgent",
        "env": env,
        "backbone_llm": llm,
        "cache_dir": "cache",
        "log_dir": "cache",
        "debug": False,
    })

    # 4. 规划
    print("[4/4] 规划中（首次调用模型，约 30-60 秒）...\n")
    extracted = llm.nl_to_python_direct(
        natural_language=QUERY,
        start_city=None, target_city=None, days=None, people_number=None,
    )

    query = {
        "uid": "demo_001",
        "nature_language": QUERY,
        "start_city": extracted["start_city"],
        "target_city": extracted["target_city"],
        "days": extracted["days"],
        "people_number": extracted["people_number"],
        # 已经生成好的约束，Agent 见到就不会再调一次模型
        "hard_logic_py": extracted["hard_logic_py"],
    }

    print(f"查询: {QUERY}")
    print(f"路线: {query['start_city']} -> {query['target_city']} | "
          f"{query['days']} 天 | {query['people_number']} 人")
    print(f"生成约束 {len(query['hard_logic_py'])} 条\n")

    # agent.run 会替换全局 stdout 做日志重定向，这里存一份用完还原
    saved = (sys.stdout, sys.stderr)
    try:
        succ, plan = agent.run(query, prob_idx=query["uid"], oralce_translation=False)
    finally:
        sys.stdout, sys.stderr = saved

    # 5. 输出
    itinerary = plan.get("itinerary") or []
    if not itinerary:
        print("未能生成有效行程。")
        return

    print(f"=== 行程（满足全部约束: {succ}）===")
    for day in itinerary:
        print(f"\n--- Day {day['day']} ---")
        for act in day["activities"]:
            pos = act.get("position") or act.get("start") or ""
            cost = act.get("cost")
            cost_str = f"  ¥{cost}" if cost is not None else ""
            print(f"  {act.get('start_time', '  :  ')}-{act.get('end_time', '  :  ')}  "
                  f"{act.get('type', ''):<13}{pos}{cost_str}")

    total = sum(a.get("cost", 0) or 0 for d in itinerary for a in d["activities"])
    print(f"\n总花费: ¥{total:.0f}")


if __name__ == "__main__":
    main()
