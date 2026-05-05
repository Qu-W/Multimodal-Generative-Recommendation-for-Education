"""最小化测试服务器 - 无任何依赖"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return "<h1>EduRec 服务器正常运行！</h1><p>如果你能看到这个页面，服务器工作正常。</p>"

@app.get("/ping")
def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Test server starting at http://127.0.0.1:7860")
    config = uvicorn.Config(app, host="127.0.0.1", port=7860, log_level="info", loop="asyncio")
    server = uvicorn.Server(config)
    server.run()
