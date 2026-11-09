"""Ray job entrypoint: run one command on a worker that holds the requested resources, streaming its output.

The job driver runs on the (GPU-less) head; this hops onto a worker. Args: <num_gpus> <num_cpus> <threads> <cmd...>.
`pytest ...` / `python ...` run with the worker's interpreter. Output lines are printed as they arrive: Ray forwards a
task's stdout to the driver, so `ray job submit` (and `ray job logs --follow`) show progress live.

Each output line is stamped with seconds since the command started (`[   45s] ...`); the start line carries the wall
clock. Every HEARTBEAT_S without output a `[ 1800s] (heartbeat) ...` line reports the command's CPU use, so a quiet log
means the command is stuck, not lost. Heartbeats are not progress: watch.py shows them but does not count them.
"""

import sys

import ray

HEARTBEAT_S = 300

num_gpus, num_cpus, threads = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
cmd = sys.argv[4:]
ray.init()


# Never on the head: it has no GPU and its CPUs belong to the driver/GCS.
@ray.remote(
    num_gpus=num_gpus, num_cpus=num_cpus, label_selector={"ray.io/node-group": "!headgroup"}
)
def run(cmd: list[str]) -> int:
    import os
    import socket
    import subprocess
    import threading
    import time

    def tree_cpu_s(root: int) -> float:
        # Live process tree only (exited children are in their parent's cutime); Ray actors it starts are not in it.
        stats = {}
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
            except OSError:
                continue
            stats[int(pid)] = (int(fields[1]), sum(map(int, fields[11:15])))
        children: dict[int, list[int]] = {}
        for pid, (ppid, _) in stats.items():
            children.setdefault(ppid, []).append(pid)
        todo, ticks = [root], 0
        while todo:
            pid = todo.pop()
            ticks += stats.get(pid, (0, 0))[1]
            todo += children.get(pid, [])
        return ticks / os.sysconf("SC_CLK_TCK")

    def heartbeat() -> None:
        cpu, since = tree_cpu_s(p.pid), time.monotonic()
        while p.poll() is None:
            # due HEARTBEAT_S after the later of the last output and the last heartbeat
            if (wait := HEARTBEAT_S - (time.monotonic() - max(last_output[0], since))) > 0:
                time.sleep(wait)
                continue
            now, now_cpu = time.monotonic(), tree_cpu_s(p.pid)
            print(
                f"[{now - start:7.0f}s] (heartbeat) no output for {(now - last_output[0]) / 60:.0f} min; command "
                f"used {now_cpu - cpu:.0f} CPU-s in the last {(now - since) / 60:.0f} min\n",
                end="",
                flush=True,
            )
            cpu, since = now_cpu, now

    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
    )
    if cmd[0] == "pytest":
        cmd = [sys.executable, "-m", "pytest", *cmd[1:]]
    elif cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    start = time.monotonic()
    last_output = [start]
    wall = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[worker {socket.gethostname()} {wall}] {' '.join(cmd)[:300]!r}", flush=True)
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1
    ) as p:
        threading.Thread(target=heartbeat, daemon=True).start()
        for line in p.stdout:
            last_output[0] = time.monotonic()
            print(f"[{last_output[0] - start:7.0f}s] {line}", end="", flush=True)
    print(f"[worker exit {p.returncode} after {time.monotonic() - start:.0f}s]", flush=True)
    return p.returncode


sys.exit(ray.get(run.remote(cmd)))
