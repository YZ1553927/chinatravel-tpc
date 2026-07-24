# Attribution

## ChinaTravel Framework

The `chinatravel/` package in this repository is based on the **ChinaTravel** neuro-symbolic AI research project developed by the **LAMDA-NeSy** group at Nanjing University.

### Original Project

- **Repository**: [LAMDA-NeSy/ChinaTravel](https://github.com/LAMDA-NeSy/ChinaTravel) (if publicly available)
- **Dataset**: `LAMDA-NeSy/ChinaTravel` on Hugging Face Datasets
- **Purpose**: Research framework for automated travel itinerary planning using Neuro-Symbolic AI

### Components Used

The following framework components are used by TPCAgent:

| Module | Purpose |
|--------|---------|
| `chinatravel/environment/tools/` | Database APIs for attractions, restaurants, hotels, transport |
| `chinatravel/symbol_verification/` | Symbolic constraint verification engine |
| `chinatravel/agent/base.py` | Agent abstract base classes |
| `chinatravel/agent/llms.py` | LLM backend implementations |
| `chinatravel/agent/nesy_agent/` | NL2SL translation and AST checking |
| `chinatravel/agent/nesy_verifier/` | Commonsense and personal constraint verification |
| `chinatravel/data/` | Dataset loading utilities |
| `chinatravel/evaluation/` | Evaluation metric utilities |

### Modifications

Minor modifications were made to these framework files to make them work as a standalone package:
- Database relative paths adjusted for package structure
- Sys.path hacks removed
- Some import paths normalized

### License

The original ChinaTravel framework code remains under its original license. This repository's TPCAgent implementation is provided under the MIT License.
