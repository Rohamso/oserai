"""Chaos testing helper: kill a random live agent container mid-batch to
demonstrate the coordinator's requeue-on-churn behavior.

This only works against the docker-compose demo, where agent container names
match their registered node names. There's no way to remotely kill a real
contributor's process -- nodes only ever pull from the coordinator, it never
reaches into them -- so on real hardware, just Ctrl-C or `kill` the agent
process yourself mid-batch and watch the dashboard: the node should flip to
offline within heartbeat_timeout_s (default 90s) and its in-flight job
should reappear as queued with an incremented retry count.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys

import httpx


def pick_target(coordinator_url: str) -> str | None:
    resp = httpx.get(f"{coordinator_url}/nodes", timeout=10)
    resp.raise_for_status()
    live = [n for n in resp.json() if n["status"] in ("idle", "busy")]
    if not live:
        return None
    return random.choice(live)["name"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coordinator-url", default="http://localhost:8000")
    parser.add_argument("--compose-file", default="docker-compose.yaml")
    parser.add_argument("--list", action="store_true", help="print live nodes and exit")
    args = parser.parse_args()

    if args.list:
        resp = httpx.get(f"{args.coordinator_url}/nodes", timeout=10)
        resp.raise_for_status()
        for n in resp.json():
            print(f"{n['name']:20} status={n['status']:8} jobs_done={n['jobs_done']}")
        return

    target = pick_target(args.coordinator_url)
    if target is None:
        print("no live nodes to kill")
        sys.exit(1)

    print(f"killing container for node '{target}' via docker compose...")
    subprocess.run(["docker", "compose", "-f", args.compose_file, "kill", target], check=False)
    print(f"done. node '{target}' should go offline within ~90s; its in-flight job should requeue.")


if __name__ == "__main__":
    main()
