"""Silent server (headers never sent): does read_timeout=5 stretch under SIGSTOP?"""
import subprocess, sys, time, os, signal, http.server, threading

class Silent(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(120)
    def log_message(self, *a): pass

srv = http.server.ThreadingHTTPServer(("127.0.0.1", 18744), Silent)
srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()

child = subprocess.Popen([sys.executable, "-c", """
import requests, time
t0=time.time()
try:
    requests.post("http://127.0.0.1:18744/", data=b"{}", timeout=(5,5))
    print("COMPLETED (unexpected)", flush=True)
except Exception as e:
    print(f"EXCEPTION {type(e).__name__} at wall {time.time()-t0:.1f}s", flush=True)
"""])
time.sleep(1)
os.kill(child.pid, signal.SIGSTOP)
print("frozen at wall 1s, holding 20s", flush=True)
time.sleep(20)
os.kill(child.pid, signal.SIGCONT)
print("resumed at wall 21s", flush=True)
child.wait(timeout=30)
