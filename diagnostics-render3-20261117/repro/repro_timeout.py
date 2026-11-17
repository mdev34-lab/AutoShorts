"""R1: requests timeout=5 is NOT a total deadline against a trickling response."""
import time, requests
from trickle_server import serve
serve()
t0 = time.time()
try:
    r = requests.get("http://127.0.0.1:18741/", timeout=5)
    print(f"R1 RESULT: request COMPLETED in {time.time()-t0:.1f}s despite timeout=5 (status {r.status_code})")
except requests.exceptions.Timeout as e:
    print(f"R1 RESULT: timed out after {time.time()-t0:.1f}s: {e}")
