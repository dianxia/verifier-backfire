# -*- coding: utf-8 -*-
"""Experiment driver: arms x HumanEval, thread pool, jsonl checkpointing.

Usage:
    python -m src.run_exp --arms all --limit 164 --threads 8
    python -m src.run_exp --arms none,oracle_full --limit 3   # smoke
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .loop import ARMS, run_gvr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "humaneval.jsonl")


def load_tasks(limit: int) -> list[dict]:
    tasks = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            tasks.append(json.loads(line))
            if len(tasks) >= limit:
                break
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms",
        default="all",
        help="Comma list or 'all'",
    )
    ap.add_argument("--limit", type=int, default=164)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--out",
        default=os.path.join(_ROOT, "results", "e1.jsonl"),
    )
    args = ap.parse_args()

    arm_names = (
        list(ARMS.keys()) if args.arms == "all"
        else [a.strip() for a in args.arms.split(",") if a.strip()]
    )
    tasks = load_tasks(args.limit)

    # resume: skip (arm, task_id) already in the checkpoint file
    done = set()
    try:
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["arm"], r["task_id"]))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass

    jobs = [
        (task, arm)
        for arm in arm_names
        for task in tasks
        if (arm, task["task_id"]) not in done
    ]
    print(
        f"arms={arm_names} tasks={len(tasks)} "
        f"jobs={len(jobs)} (skipped {len(done)} done)",
        flush=True,
    )

    lock = threading.Lock()
    counter = {"done": 0, "tokens": 0}
    t0 = time.time()

    def worker(task: dict, arm: str) -> dict:
        return run_gvr(task, arm)

    with open(args.out, "a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = {
                ex.submit(worker, t, a): (t, a) for t, a in jobs
            }
            for fut in as_completed(futures):
                task, arm = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {
                        "arm": arm,
                        "task_id": task["task_id"],
                        "error": repr(e)[:500],
                    }
                with lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    counter["done"] += 1
                    if counter["done"] % 20 == 0:
                        el = time.time() - t0
                        print(
                            f"[{counter['done']}/{len(jobs)}] "
                            f"{el:.0f}s elapsed, "
                            f"~{el / max(counter['done'], 1):.1f}s/job",
                            flush=True,
                        )
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()