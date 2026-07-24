# Authors

## TPCAgent Implementation

**YZ1553927** ([@YZ1553927](https://github.com/YZ1553927))

The following files are the core contribution of this repository:

| File | Lines | Description |
|------|-------|-------------|
| `tpc_agent/agent.py` | 3,651 | DFS-based itinerary planning engine with symbolic constraint verification |
| `tpc_agent/llm.py` | 1,116 | Natural language to symbolic constraint translation via LLM |
| `tpc_agent/converter.py` | 547 | Structured JSON extraction from natural language queries |
| `tpc_agent/utils.py` | 123 | Time/cost calculation utilities and MMR algorithm |

Total: **5,437 lines** of original code.

## ChinaTravel Framework

The `chinatravel/` package (environment simulation, symbolic verification, evaluation tools) was developed by the **LAMDA-NeSy** research group at Nanjing University.

See [ATTRIBUTION.md](ATTRIBUTION.md) for details.
