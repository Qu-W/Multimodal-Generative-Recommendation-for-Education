"""
server.py  —  EduRec HTTP 服务（标准库 http.server，无需 uvicorn/fastapi）
运行: python server.py
访问: http://127.0.0.1:7860
"""
import json
import os
import sys
import yaml
import urllib.parse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.faiss_retriever import FaissRetriever
from src.retrieval.bpr_retriever import BPRRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LLMClient
from src.generation.generator import Generator
from src.reranking.reranker import LLMReranker

# ── 全局初始化 ─────────────────────────────────────────────────────────────────

with open("configs/config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

PROCESSED_DIR = CFG["data"]["processed_dir"]
INDEX_DIR     = CFG["data"]["index_dir"]

print("Loading courses.json ...")
with open(Path(PROCESSED_DIR) / "courses.json", encoding="utf-8") as f:
    COURSES: dict = json.load(f)
for c in COURSES.values():
    if "about" not in c:
        c["about"] = c.get("description", "") or " ".join(c.get("concepts", [])[:10])

print("Loading user_sequences.json ...")
with open(Path(PROCESSED_DIR) / "user_sequences.json", encoding="utf-8") as f:
    USER_SEQ: dict = json.load(f)

print("Loading test.json ...")
with open(Path(PROCESSED_DIR) / "test.json", encoding="utf-8") as f:
    TEST_DATA: dict = json.load(f)

print("Initializing retrievers ...")
faiss_ret = FaissRetriever(INDEX_DIR, {**CFG["embedding"], **CFG["retrieval"]})
bpr_ret   = BPRRetriever(PROCESSED_DIR, CFG["retrieval"])

_index_ready = Path(INDEX_DIR, "course.faiss").exists()
if _index_ready:
    faiss_ret.load(COURSES)
    bpr_ret.load(COURSES)
    print("FAISS index loaded (BGE-M3 will load on first query)")
else:
    print("WARNING: FAISS index not found")

hybrid   = HybridRetriever(faiss_ret, bpr_ret, alpha=0.7)
llm      = LLMClient(CFG["generation"])
gen      = Generator(llm, {"max_history": 10, "max_candidates": 8})
reranker = LLMReranker(gen, top_k=CFG["retrieval"]["top_k_rerank"])

SAMPLE_USERS = list(TEST_DATA.keys())[:50]

# ── 业务逻辑 ───────────────────────────────────────────────────────────────────

def get_user_history(user_id: str):
    seq = USER_SEQ.get(user_id, [])
    history_ids = seq[:-2] if len(seq) > 2 else seq[:-1] if len(seq) > 1 else []
    return [COURSES[cid] for cid in history_ids if cid in COURSES]


def do_recommend(user_id: str, user_query: str, use_llm: bool):
    if not _index_ready:
        return {"error": "索引未构建，请先运行 python scripts/build_index.py"}

    user_history = get_user_history(user_id.strip())
    query = user_query.strip() or " ".join(c.get("name", "") for c in user_history[-3:])
    if not query:
        query = "课程推荐"

    candidates = hybrid.retrieve(
        query=query,
        user_id=user_id.strip() or None,
        exclude_ids=[c["course_id"] for c in user_history],
        top_k=CFG["retrieval"]["top_k_recall"],
    )

    if not candidates:
        return {"error": "未找到相关课程，请检查索引或调整查询"}

    if use_llm:
        results = reranker.rerank(user_history, candidates, user_query=user_query)
    else:
        results = candidates[:CFG["retrieval"]["top_k_rerank"]]

    return {
        "history": [
            {"course_id": c["course_id"], "name": c.get("name", ""), "about": c.get("about", "")}
            for c in user_history[-5:]
        ],
        "results": [
            {
                "course_id": c["course_id"],
                "name": c.get("name", c["course_id"]),
                "about": c.get("about", c.get("description", ""))[:200],
                "score": round(float(c.get("hybrid_score", c.get("score", 0.0))), 4),
                "reason": c.get("reason", ""),
            }
            for c in results
        ],
    }

# ── HTTP 请求处理 ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"  {self.address_string()} - {format % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))

        if parsed.path == "/":
            self.send_html(build_html())

        elif parsed.path == "/recommend":
            user_id  = params.get("user_id", "").strip()
            query    = params.get("query", "").strip()
            use_llm  = params.get("use_llm", "false").lower() == "true"
            if not user_id:
                self.send_json({"error": "请提供 user_id"}, 400)
            else:
                result = do_recommend(user_id, query, use_llm)
                self.send_json(result)

        elif parsed.path == "/users":
            self.send_json(SAMPLE_USERS)

        else:
            self.send_response(404)
            self.end_headers()

# ── HTML 前端 ──────────────────────────────────────────────────────────────────

def build_html():
    users_json = json.dumps(SAMPLE_USERS, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>EduRec 教育课程推荐</title>
<style>
  body {{ font-family: "Microsoft YaHei", sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  .panel {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  label {{ font-weight: bold; display: block; margin-bottom: 6px; color: #555; }}
  select, input[type=text] {{ width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; box-sizing: border-box; }}
  button {{ background: #3498db; color: white; border: none; padding: 10px 28px; border-radius: 4px; cursor: pointer; font-size: 15px; margin-top: 16px; }}
  button:hover {{ background: #2980b9; }}
  button:disabled {{ background: #aaa; cursor: not-allowed; }}
  .card {{ border-left: 4px solid #3498db; padding: 12px 16px; margin: 10px 0; background: #f9fbff; border-radius: 4px; }}
  .card.history {{ border-left-color: #27ae60; background: #f9fff9; }}
  .card h3 {{ margin: 0 0 6px; font-size: 15px; color: #2c3e50; }}
  .card p {{ margin: 4px 0; font-size: 13px; color: #666; }}
  .score {{ display: inline-block; background: #e8f4fd; color: #2980b9; padding: 2px 8px; border-radius: 10px; font-size: 12px; margin-left: 8px; }}
  .reason {{ color: #8e44ad; font-style: italic; margin-top: 6px; font-size: 13px; }}
  #status {{ margin-left: 16px; font-weight: bold; }}
  .section-title {{ font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px; }}
</style>
</head>
<body>
<h1>🎓 EduRec 教育课程推荐系统</h1>
<p style="color:#888">基于 MOOCCube · 语义检索 + 协同过滤 + LLM 重排</p>

<div class="panel">
  <label>选择用户 ID</label>
  <select id="user_sel" onchange="document.getElementById('user_input').value=this.value">
    <option value="">-- 从样本用户中选择 --</option>
  </select>
  <input type="text" id="user_input" placeholder="或手动输入用户 ID" style="margin-top:8px">

  <label style="margin-top:16px">补充需求（可选）</label>
  <input type="text" id="query" placeholder="例如：我想学深度学习">

  <div style="margin-top:12px">
    <input type="checkbox" id="use_llm">
    <label style="display:inline;font-weight:normal"> 启用 LLM 重排（需要 API Key）</label>
  </div>

  <button id="btn" onclick="recommend()">🔍 推荐</button>
  <span id="status"></span>
</div>

<div id="result_area" style="display:none">
  <div class="panel">
    <div class="section-title">📚 用户学习历史（最近 5 门）</div>
    <div id="history_out"></div>
  </div>
  <div class="panel">
    <div class="section-title">✨ 推荐结果</div>
    <div id="results_out"></div>
  </div>
</div>

<script>
const users = {users_json};
const sel = document.getElementById('user_sel');
users.forEach(u => {{
  const o = document.createElement('option');
  o.value = o.textContent = u;
  sel.appendChild(o);
}});

async function recommend() {{
  const uid = document.getElementById('user_input').value.trim();
  const query = document.getElementById('query').value.trim();
  const useLlm = document.getElementById('use_llm').checked;
  const btn = document.getElementById('btn');
  const status = document.getElementById('status');

  if (!uid) {{ alert('请输入或选择用户 ID'); return; }}
  btn.disabled = true;
  status.style.color = '#3498db';
  status.textContent = '⏳ 推荐中' + (useLlm ? '（LLM重排，约30-60s）' : '（首次加载模型约30-60s）') + '...';

  try {{
    const url = '/recommend?user_id=' + encodeURIComponent(uid) +
                '&query=' + encodeURIComponent(query) +
                '&use_llm=' + useLlm;
    const res = await fetch(url);
    const data = await res.json();

    if (data.error) {{
      status.style.color = '#e74c3c';
      status.textContent = '❌ ' + data.error;
      return;
    }}

    renderHistory(data.history || []);
    renderResults(data.results || []);
    document.getElementById('result_area').style.display = 'block';
    status.style.color = '#27ae60';
    status.textContent = '✅ 完成（' + (data.results||[]).length + ' 条推荐）';
  }} catch(e) {{
    status.style.color = '#e74c3c';
    status.textContent = '❌ 请求失败: ' + e.message;
  }} finally {{
    btn.disabled = false;
  }}
}}

function renderHistory(list) {{
  const el = document.getElementById('history_out');
  if (!list.length) {{ el.innerHTML = '<p style="color:#aaa">（无历史记录）</p>'; return; }}
  el.innerHTML = list.map((c,i) => `<div class="card history">
    <h3>${{i+1}}. ${{esc(c.name||c.course_id)}}</h3>
    <p>${{esc((c.about||'').slice(0,100))}}</p></div>`).join('');
}}

function renderResults(list) {{
  const el = document.getElementById('results_out');
  if (!list.length) {{ el.innerHTML = '<p style="color:#aaa">（无结果）</p>'; return; }}
  el.innerHTML = list.map((c,i) => `<div class="card">
    <h3>${{i+1}}. ${{esc(c.name||c.course_id)}} <span class="score">score: ${{c.score}}</span></h3>
    <p>${{esc((c.about||'').slice(0,150))}}</p>
    ${{c.reason ? '<p class="reason">💡 ' + esc(c.reason) + '</p>' : ''}}</div>`).join('');
}}

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
</script>
</body>
</html>"""

# ── 启动 ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    host, port = "127.0.0.1", 7860
    httpd = HTTPServer((host, port), Handler)
    print(f"\n✅ EduRec running at http://{host}:{port}")
    print("   按 Ctrl+C 停止\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        httpd.server_close()
