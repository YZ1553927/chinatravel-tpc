"""部署级配置加载 —— 从仓库根的 .env 读取环境变量，无第三方依赖。

被 converter.py 和 llm.py 共用，保证无论谁先被构造，配置都已就绪。
"""

import os

# 本文件位于 <repo>/tpc_agent/_config.py，向上两级即仓库根
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_PATH = os.path.join(_REPO_ROOT, ".env")

_loaded = False


def load_dotenv(force: bool = False) -> None:
    """把 .env 里的 KEY=VALUE 注入 os.environ。

    已存在的环境变量优先，不会被 .env 覆盖（方便临时用 shell 变量压过配置）。
    支持 # 注释、空行、单双引号包裹的值。
    """
    global _loaded
    if _loaded and not force:
        return
    _loaded = True

    if not os.path.isfile(ENV_PATH):
        return
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.split(" #")[0].strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"⚠️ 读取 .env 失败 ({ENV_PATH}): {e}")


def env(key: str, default: str = None) -> str:
    """读配置：先确保 .env 已加载，再取环境变量。"""
    load_dotenv()
    return os.environ.get(key, default)
