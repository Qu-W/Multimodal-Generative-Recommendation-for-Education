"""纯标准库 HTTP 服务器测试"""
from http.server import HTTPServer, BaseHTTPRequestHandler

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<h1>OK - server works!</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, fmt, *args):
        print("request:", fmt % args)

for host in ["127.0.0.1", "0.0.0.0", "localhost"]:
    try:
        httpd = HTTPServer((host, 7861), H)
        print(f"Bound to {host}:7861 - try http://127.0.0.1:7861")
        httpd.serve_forever()
        break
    except Exception as e:
        print(f"Failed {host}: {e}")
