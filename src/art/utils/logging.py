import time


# ---------- lightweight "nice print" helpers ----------
class _C:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    ITAL = "\x1b[3m"
    GRAY = "\x1b[90m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    MAGENTA = "\x1b[35m"


def _ts():
    return time.strftime("%H:%M:%S")


def info(msg):
    print(f"[{_ts()}] {_C.BLUE}INFO{_C.RESET}  {msg}")


def step(msg):
    print(f"[{_ts()}] {_C.CYAN}STEP{_C.RESET}  {msg}")


def ok(msg):
    print(f"[{_ts()}] {_C.GREEN}OK{_C.RESET}    {msg}")


def warn(msg):
    print(f"[{_ts()}] {_C.YELLOW}WARN{_C.RESET}  {msg}")


def err(msg):
    print(f"[{_ts()}] {_C.RED}ERR{_C.RESET}   {msg}")


def dim(msg):
    print(f"{_C.DIM}{msg}{_C.RESET}")
