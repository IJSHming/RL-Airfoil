"""HPC 任务 5 小时运行窗口监控与倒计时跟踪器。
提供实时剩余时间查询与安全裕度预警。
"""

import time
import os
import sys

TOTAL_BUDGET_SECONDS = 5 * 3600  # 5 小时安全裕量
START_TIME_FILE = ".hpc_timer_start.txt"


def init_timer():
    if not os.path.exists(START_TIME_FILE):
        with open(START_TIME_FILE, "w") as f:
            f.write(str(time.time()))


def get_status():
    if not os.path.exists(START_TIME_FILE):
        init_timer()
    with open(START_TIME_FILE, "r") as f:
        start_t = float(f.read().strip())
    elapsed = time.time() - start_t
    remaining = max(0.0, TOTAL_BUDGET_SECONDS - elapsed)
    return {
        "start_time": start_t,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60.0,
        "remaining_sec": remaining,
        "remaining_min": remaining / 60.0,
        "remaining_hours": remaining / 3600.0,
        "pct_used": (elapsed / TOTAL_BUDGET_SECONDS) * 100.0,
    }


def print_status():
    s = get_status()
    print(f"[HPC Timer] 已运行: {s['elapsed_min']:.1f} 分钟 | 剩余安全时间: {s['remaining_hours']:.2f} 小时 ({s['remaining_min']:.1f} 分钟) | 时间消耗: {s['pct_used']:.1f}%")


if __name__ == "__main__":
    init_timer()
    print_status()
