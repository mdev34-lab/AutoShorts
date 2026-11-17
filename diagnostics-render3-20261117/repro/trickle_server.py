"""Local trickle server: sends one byte every INTERVAL seconds, forever-ish."""
import http.server, threading

INTERVAL = 2.0

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        import time
        for _ in range(10):
            try:
                self.wfile.write(b" ")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(INTERVAL)
    def do_POST(self):
        self.do_GET()
    def log_message(self, *a):
        pass

def serve(port=18741):
    srv = http.server.HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
