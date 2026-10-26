"""
简单的旅行规划后端 —— 用中文直接提问，返回完整行程 JSON。

启动:
    python server.py                          # http://127.0.0.1:8000
    python server.py --host 0.0.0.0 --port 8080

依赖:
    pip install fastapi "uvicorn[standard]"

配置（写在项目根的 .env 里，或设成环境变量）:
    TPC_API_KEY     必填，除非要用本地 Qwen 模型
    TPC_API_BASE    默认 https://api.deepseek.com
    TPC_API_MODEL   默认 deepseek-chat

    换厂商只需改后两项，例如智谱:
        TPC_API_BASE=https://open.bigmodel.cn/api/paas/v4
        TPC_API_MODEL=glm-4-plus

用法:
    浏览器打开 http://127.0.0.1:8000 有输入框可以直接问。
    命令行:
        curl -X POST localhost:8000/plan -H "Content-Type: application/json" \
             -d '{"query":"我们一家三口从成都去北京玩5天，预算1万，住大床房"}'
        curl localhost:8000/plan/<job_id>
"""

import argparse
import os
import queue
import sys
import threading
import time
import traceback
import uuid

project_root_path = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(project_root_path, "chinatravel"), project_root_path):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────
# 规划器：单例，封装 LLM + TPCAgent
# ──────────────────────────────────────────────────────────────────────


class Planner:
    """把「一句中文」变成「一份行程」。

    模型和 agent 只在第一次请求时初始化（懒加载），避免导入模块就吃掉几 GB 显存。
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.llm = None
        self.agent = None
        self.init_error = None
        self._lock = threading.Lock()
        self._done = False

    # -- 初始化 ---------------------------------------------------------
    def ensure_ready(self):
        if self._done:
            if self.init_error:
                raise RuntimeError(self.init_error)
            return

        with self._lock:
            if self._done:
                if self.init_error:
                    raise RuntimeError(self.init_error)
                return
            try:
                from chinatravel.agent.load_model import init_agent, init_llm
                from chinatravel.environment.world_env import WorldEnv

                log_dir = os.path.join(project_root_path, "cache", "server")
                os.makedirs(log_dir, exist_ok=True)

                self.llm = init_llm("TPCLLM")

                converter = getattr(self.llm, "travel_converter", None)
                if converter is None:
                    raise RuntimeError(
                        "TPCLLM 没有可用后端：既没有 API key，也没找到本地模型。\n"
                        "  方案一：在 .env 里设置 TPC_API_KEY\n"
                        "  方案二：把模型放到 chinatravel/local_llm/Qwen3-4B/"
                    )

                self.agent = init_agent({
                    "method": "TPCAgent",
                    "env": WorldEnv(),
                    "backbone_llm": self.llm,
                    "cache_dir": os.path.join(project_root_path, "cache"),
                    "log_dir": log_dir,
                    "debug": self.debug,
                })
                print(f"✅ 规划器就绪，后端 = {converter.name}")
            except Exception as e:
                self.init_error = f"{type(e).__name__}: {e}"
            finally:
                self._done = True

        if self.init_error:
            raise RuntimeError(self.init_error)

    # -- 单次规划 -------------------------------------------------------
    def plan(self, nl: str, job_id: str):
        self.ensure_ready()
        converter = self.llm.travel_converter

        # 1) 先抽取一次。既用来校验关键事实，也复用给 DSL 生成，避免同一句话调两次模型。
        raw = converter.convert_with_fallback(nl)
        pretty = converter.get_pretty_data(raw)

        # 抽不到就直接报错，不要让它静默套用"北京→上海/3天"的默认值，
        # 那样用户会拿到一份完全跑偏的行程却看不出问题。
        missing = [
            label for label, key in
            [("出发城市", "start_city"), ("目标城市", "target_city"),
             ("天数", "travel_days"), ("人数", "people_count")]
            if not pretty.get(key)
        ]
        if missing:
            raise ValueError(
                f"没能从这句话里识别出：{'、'.join(missing)}。\n"
                "请说得更明确一些，例如：『我们一家三口从成都去北京玩5天，预算1万，住大床房』"
            )

        # 2) 生成约束 DSL（复用上一步的抽取结果）
        extracted = self.llm.nl_to_python_direct(
            natural_language=nl,
            start_city=pretty["start_city"],
            target_city=pretty["target_city"],
            days=pretty["travel_days"],
            people_number=pretty["people_count"],
            structured_data=pretty,
        )

        query = {
            "uid": job_id,
            "nature_language": nl,
            "start_city": extracted.get("start_city"),
            "target_city": extracted.get("target_city"),
            "days": extracted.get("days"),
            "people_number": extracted.get("people_number"),
            # 已经生成好了，TPCAgent 见到它就不会再调一次模型
            "hard_logic_py": extracted.get("hard_logic_py", []),
        }

        # 3) 符号搜索
        # TPCAgent.run 内部会替换全局的 sys.stdout/stderr 去做日志重定向，
        # 这里存一份、用完还原，避免污染整个进程。
        saved = (sys.stdout, sys.stderr)
        try:
            succ, plan = self.agent.run(query, prob_idx=job_id, oralce_translation=False)
        finally:
            sys.stdout, sys.stderr = saved

        return succ, plan, extracted


# ──────────────────────────────────────────────────────────────────────
# 任务队列：TPCAgent 会改全局 stdout、WorldEnv 是有状态的，
# 所以请求必须串行执行，不能并发。
# ──────────────────────────────────────────────────────────────────────


def _jsonable(obj):
    """递归把 numpy / torch 标量等转成原生类型。

    行程里的 price/tickets 等字段是 numpy.float64 / numpy.int64，
    FastAPI 的 jsonable_encoder 处理不了，会抛
    "TypeError: 'numpy.int64' object is not iterable"。
    项目里的 NpEncoder 只对 json.dump 生效，这里需要自己兜一层。
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        return obj
    # numpy / torch 标量都带 .item()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    # 数组类带 .tolist()
    if hasattr(obj, "tolist"):
        try:
            return _jsonable(obj.tolist())
        except Exception:
            pass
    return str(obj)


class Job:
    def __init__(self, nl: str):
        self.id = uuid.uuid4().hex[:12]
        self.query = nl
        # queued | running | done（满足全部约束）| partial（有方案但约束没满足）| failed
        self.status = "queued"
        self.result = None
        self.extracted = None
        self.error = None
        self.satisfied = None
        self.created = time.time()
        self.finished = None

    def to_dict(self, with_result: bool = True):
        d = {
            "job_id": self.id,
            "status": self.status,
            "query": self.query,
            "created": self.created,
        }
        if self.finished:
            d["elapsed_sec"] = round(self.finished - self.created, 1)
        if self.status == "failed":
            d["error"] = self.error
        if with_result and self.status in ("done", "partial"):
            d["satisfied"] = bool(self.satisfied)
            d["constraints"] = _jsonable((self.extracted or {}).get("hard_logic_py", []))
            d["plan"] = _jsonable(self.result)
        return d


_jobs: dict = {}
_jobs_lock = threading.Lock()
_job_queue: "queue.Queue[Job]" = queue.Queue()
_planner = Planner()


def _worker():
    while True:
        job = _job_queue.get()
        if job is None:
            break
        job.status = "running"
        try:
            succ, plan, extracted = _planner.plan(job.query, job.id)
            job.result = plan
            job.extracted = extracted
            job.satisfied = bool(succ)
            # succ=False 表示搜索没找到满足全部约束的方案，返回的是兜底结果。
            # 不能和成功混为一谈 —— 否则用户拿到一份"看着像行程"的东西，
            # 实际却违反了明确提出的要求（比如"想去故宫"根本没去）。
            job.status = "done" if succ else "partial"
            print(f"{'✅' if succ else '⚠️'} [{job.id}] {job.status}")
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
            job.status = "failed"
            print(f"❌ [{job.id}] 失败: {job.error}")
            if _planner.debug:
                traceback.print_exc()
        finally:
            job.finished = time.time()
            _job_queue.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


# ──────────────────────────────────────────────────────────────────────
# HTTP 接口
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="ChinaTravel 规划后端", version="1.0")


class PlanRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    """不触发模型加载，只报告配置状态。"""
    # 懒加载之前也要能读到 .env，否则这里会误报"没配 key"
    try:
        from chinatravel.agent.tpc_agent._config import load_dotenv
        load_dotenv()
    except Exception:
        pass

    key_vars = ("TPC_API_KEY", "DEEPSEEK_API_KEY", "ZHIPUAI_API_KEY",
                "GLM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY")
    has_key = any(os.environ.get(k) for k in key_vars)
    local_model = os.path.isdir(
        os.path.join(project_root_path, "chinatravel", "local_llm", "Qwen3-4B")
    )
    return {
        "status": "ok",
        "ready": _planner._done and not _planner.init_error,
        "init_error": _planner.init_error,
        "api_key_set": has_key,
        "local_model_found": local_model,
        "base_url": os.environ.get("TPC_API_BASE", "https://api.deepseek.com"),
        "model": os.environ.get("TPC_API_MODEL", "deepseek-chat"),
        "queue_depth": _job_queue.qsize(),
    }


@app.post("/plan", status_code=202)
def submit(req: PlanRequest):
    """提交一句中文需求，立即返回 job_id；用 GET /plan/{job_id} 取结果。"""
    nl = (req.query or "").strip()
    if not nl:
        raise HTTPException(status_code=400, detail="query 不能为空")
    if len(nl) > 2000:
        raise HTTPException(status_code=400, detail="query 过长（上限 2000 字）")

    job = Job(nl)
    with _jobs_lock:
        _jobs[job.id] = job
    _job_queue.put(job)
    return {"job_id": job.id, "status": job.status,
            "hint": f"GET /plan/{job.id} 查询结果"}


@app.get("/plan/{job_id}")
def get_plan(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"没有这个任务: {job_id}")
    return job.to_dict()


@app.get("/jobs")
def list_jobs(limit: int = 20):
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.created, reverse=True)[:limit]
    return [j.to_dict(with_result=False) for j in jobs]


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>旅行规划</title>
<style>
 body{font-family:-apple-system,"Microsoft YaHei",sans-serif;max-width:900px;
      margin:40px auto;padding:0 20px;color:#222;line-height:1.6}
 h1{font-size:20px;font-weight:600}
 textarea{width:100%;height:90px;padding:10px;font-size:15px;font-family:inherit;
          border:1px solid #ccc;border-radius:6px;box-sizing:border-box;resize:vertical}
 button{margin-top:10px;padding:9px 22px;font-size:15px;border:0;border-radius:6px;
        background:#2563eb;color:#fff;cursor:pointer}
 button:disabled{background:#9aa5b1;cursor:default}
 .ex{color:#2563eb;cursor:pointer;text-decoration:underline;font-size:13px}
 #status{margin-top:14px;color:#555;font-size:14px}
 pre{background:#f6f8fa;padding:14px;border-radius:6px;overflow:auto;
     font-size:12.5px;max-height:520px;white-space:pre-wrap;word-break:break-all}
 .err{background:#fef2f2;border-left:4px solid #dc2626;padding:12px;border-radius:4px;
      white-space:pre-wrap;color:#991b1b}
</style></head><body>
<h1>🗺️ 旅行规划</h1>
<p style="color:#666;font-size:14px">用一句中文描述你的行程，系统会生成满足约束的完整方案。</p>
<textarea id="q" placeholder="例如：我们一家三口从成都去北京玩5天，预算1万元，住大床房，往返坐飞机"></textarea>
<div style="margin-top:8px">
  示例：<span class="ex">我两个人从上海去杭州玩3天，预算5000，想坐高铁，住双床房</span> ·
  <span class="ex">一家三口成都到上海5天，预算1万，想去迪士尼，市内只坐地铁</span>
</div>
<button id="go" onclick="ask()">生成行程</button>
<div id="status"></div>
<div id="out"></div>
<script>
document.querySelectorAll('.ex').forEach(e=>e.onclick=()=>{q.value=e.textContent});
async function ask(){
  const query=q.value.trim(); if(!query) return;
  go.disabled=true; out.innerHTML=''; status.textContent='提交中…';
  try{
    const r=await fetch('/plan',{method:'POST',headers:{'Content-Type':'application/json'},
                                body:JSON.stringify({query})});
    const d=await r.json();
    if(!r.ok){ throw new Error(d.detail||JSON.stringify(d)); }
    const id=d.job_id; let t0=Date.now();
    const poll=setInterval(async()=>{
      const s=await (await fetch('/plan/'+id)).json();
      const sec=((Date.now()-t0)/1000).toFixed(0);
      if(s.status==='queued') status.textContent=`排队中… ${sec}s`;
      else if(s.status==='running') status.textContent=`规划中… ${sec}s（搜索可能要几分钟）`;
      else{
        clearInterval(poll); go.disabled=false;
        if(s.status==='failed'){ status.textContent='失败';
          out.innerHTML='<div class="err">'+s.error.replace(/</g,'&lt;')+'</div>'; return; }
        status.textContent=(s.satisfied?'✅ 完成，全部约束已满足':'⚠️ 有方案，但部分约束未能满足')
          +`，用时 ${s.elapsed_sec}s`;
        out.innerHTML=(s.satisfied?'':'<div class="err" style="background:#fffbeb;'
            +'border-left-color:#f59e0b;color:#92400e">搜索没能找到满足全部约束的方案，'
            +'下面是退而求其次的结果。请检查约束是否与沙盒数据冲突（例如景点名不在库里）。</div>')
          +'<h3 style="font-size:15px">约束（'+s.constraints.length+' 条）</h3><pre>'
          + s.constraints.join('\\n\\n').replace(/</g,'&lt;') + '</pre>'
          + '<h3 style="font-size:15px">行程</h3><pre>'
          + JSON.stringify(s.plan,null,2).replace(/</g,'&lt;') + '</pre>';
      }
    },1500);
  }catch(e){ go.disabled=false; status.textContent='出错';
    out.innerHTML='<div class="err">'+String(e.message).replace(/</g,'&lt;')+'</div>'; }
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true", help="打印 agent 详细日志")
    args = ap.parse_args()

    print(f"→ http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
