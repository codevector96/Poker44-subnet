#!/usr/bin/env python3
"""Download and cache the Poker44 public training benchmark.

The benchmark API serves *raw* hands (full action sequences, fine-grained
amounts, real seat numbers). Production validators send miners hands that have
been projected through ``poker44.validator.payload_view.build_miner_payload_hand``
(action sampling, amount coarsening, seat aliasing). To avoid train/serve skew,
we cache the raw benchmark JSON here and apply that exact projection later, at
feature-extraction time.

Usage:
    python scripts/training/download_benchmark.py --out data/benchmark --max-dates 31
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_BASE = os.getenv(
    "POKER44_BENCHMARK_BASE_URL", "https://api.poker44.net/api/v1/benchmark"
)


def _get(base: str, path: str, *, timeout: int = 60, retries: int = 4) -> Any:
    url = f"{base}{path}"
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            decoded = json.loads(raw) if raw else None
            if isinstance(decoded, dict) and "data" in decoded:
                return decoded["data"]
            return decoded
        except Exception as exc:  # noqa: BLE001 - best effort with backoff
            last_exc = exc
            print(f"  ! request failed ({attempt}/{retries}): {exc}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_exc}")


def list_release_dates(base: str, max_dates: int) -> List[str]:
    payload = _get(base, f"/releases?limit={max_dates}")
    releases = payload.get("releases", []) if isinstance(payload, dict) else []
    return [str(r["sourceDate"]) for r in releases if r.get("sourceDate")]


def fetch_all_chunks_for_date(base: str, date: str, page_limit: int = 24) -> List[Dict[str, Any]]:
    """Page through every chunk-group for one release date."""
    out: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        path = f"/chunks?sourceDate={date}&limit={page_limit}"
        if cursor:
            path += f"&cursor={cursor}"
        payload = _get(base, path)
        if not isinstance(payload, dict):
            break
        chunks = payload.get("chunks", []) or []
        out.extend(chunks)
        cursor = payload.get("nextCursor")
        if not cursor or not chunks:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default="data/benchmark")
    ap.add_argument("--max-dates", type=int, default=31)
    ap.add_argument("--force", action="store_true", help="re-download cached dates")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dates = list_release_dates(args.base, args.max_dates)
    print(f"Found {len(dates)} release dates: {dates}")

    total_groups = 0
    total_batches = 0
    for date in dates:
        dest = os.path.join(args.out, f"{date}.json")
        if os.path.exists(dest) and not args.force:
            with open(dest) as fh:
                cached = json.load(fh)
            n_groups = len(cached.get("chunk_groups", []))
            n_batches = sum(len(g.get("groundTruth", [])) for g in cached.get("chunk_groups", []))
            print(f"  = {date}: cached ({n_groups} groups, {n_batches} labeled batches)")
            total_groups += n_groups
            total_batches += n_batches
            continue

        groups = fetch_all_chunks_for_date(args.base, date)
        n_batches = sum(len(g.get("groundTruth", [])) for g in groups)
        record = {"sourceDate": date, "chunk_groups": groups}
        with open(dest, "w") as fh:
            json.dump(record, fh)
        print(f"  + {date}: downloaded {len(groups)} groups, {n_batches} labeled batches")
        total_groups += len(groups)
        total_batches += n_batches

    print(
        f"\nDone. {len(dates)} dates, {total_groups} chunk-groups, "
        f"{total_batches} labeled batches cached in {args.out}/"
    )


if __name__ == "__main__":
    main()
