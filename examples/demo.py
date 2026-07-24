"""
Demo: Run TPCAgent with a sample query.

Usage:
    python examples/demo.py
"""

import sys
import os

# Ensure the repo root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tpc_agent import TPCAgent
from chinatravel.agent.load_model import init_llm


def main():
    # 1. Create LLM backend
    print("Initializing LLM...")
    llm = init_llm("deepseek")  # Change to "gpt-4o", "Qwen3-4B", etc.

    # 2. Prepare a sample query
    query = {
        "uid": "demo_001",
        "nature_language": (
            "我们一家三口从南京出发，去北京玩两天。"
            "预算3000元，想去故宫和长城，吃北京烤鸭，住经济型酒店。"
        ),
        "start_city": "南京",
        "target_city": "北京",
        "days": 2,
        "people_number": 3,
    }

    print(f"Query: {query['nature_language']}")
    print(f"From: {query['start_city']} -> To: {query['target_city']}")
    print(f"Days: {query['days']}, People: {query['people_number']}")

    # 3. Create and run the agent
    print("\nPlanning itinerary...")
    agent = TPCAgent(
        env=None,  # Replace with WorldEnv or compatible environment
        backbone_llm=llm,
        cache_dir="cache/",
        debug=False,
    )

    result = agent.run(query)

    # 4. Print results
    if result and "plan" in result:
        print(f"\n=== Generated Itinerary ({len(result['plan'])} days) ===")
        for day in result["plan"]:
            print(f"\n--- Day {day['day']} ---")
            for act in day["activities"]:
                print(f"  [{act['type']}] {act.get('position', 'N/A')}")
                print(f"    Time: {act.get('start_time', '?')} - {act.get('end_time', '?')}")
                if "cost" in act:
                    print(f"    Cost: {act['cost']} CNY")
    else:
        print("No valid itinerary found.")


if __name__ == "__main__":
    main()
