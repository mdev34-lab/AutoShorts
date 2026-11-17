"""Does a wall-clock socket timeout stretch when the process is frozen?
Request to a silent server, read_timeout=5; SIGSTOP at ~1s, hold 20s, SIGCONT.
If ReadTimeout fires only after resume, wall-clock bounds stretch under stops."""
import subprocess, sys, time, os, signal
from trickle_server import serve
serve(port=18743)

child = subprocess.Popen([sys.executable, "-c", """
import requests, time
t0=time.time()
try:
    requests.post("http://127.0.0.1:18743/", data=b"{}", timeout=(5,5))
except Exception as e:
    print(f"EXCEPTION {type(e).__name__} at wall {time.time()-t0:.1f}s", flush=True)
"""])
time.sleep(1)
os.kill(child.pid, signal.SIGSTOP)
print("frozen at wall 1s, holding 20s", flush=True)
time.sleep(20)
os.kill(child.pid, signal.SIGCONT)
print("resumed at wall 21s", flush=True)
child.wait(timeout=20)
