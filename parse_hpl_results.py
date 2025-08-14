#!/usr/bin/env python3
import re
import csv
import argparse
from pathlib import Path
from datetime import datetime

# ---------- Regex patterns ----------
RE_KV_INT = {
    "N": re.compile(r"^N\s*:\s*(\d+)"),
    "NB": re.compile(r"^NB\s*:\s*(\d+)"),
    "P": re.compile(r"^P\s*:\s*(\d+)"),
    "Q": re.compile(r"^Q\s*:\s*(\d+)"),
    "NBMIN": re.compile(r"^NBMIN\s*:\s*(\d+)"),
    "NDIV": re.compile(r"^NDIV\s*:\s*(\d+)"),
    "DEPTH": re.compile(r"^DEPTH\s*:\s*(\d+)"),
    "ALIGN_words": re.compile(r"^ALIGN\s*:\s*(\d+)"),
}
RE_KV_STR = {
    "PMAP": re.compile(r"^PMAP\s*:\s*(.+)"),
    "PFACT": re.compile(r"^PFACT\s*:\s*(.+)"),
    "RFACT": re.compile(r"^RFACT\s*:\s*(.+)"),
    "BCAST": re.compile(r"^BCAST\s*:\s*(.+)"),
    "L1": re.compile(r"^L1\s*:\s*(.+)"),
    "U": re.compile(r"^U\s*:\s*(.+)"),
    "EQUIL": re.compile(r"^EQUIL\s*:\s*(.+)"),
}
RE_SWAP = re.compile(r"^SWAP\s*:\s*(.+?)(?:\s*\(threshold\s*=\s*(\d+)\))?\s*$")

# Results line (variant, N, NB, P, Q, Time, Gflops)
RE_RESULT_ROW = re.compile(
    r"^([A-Z0-9]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+(?:e[+-]?\d+)?)\s+([0-9.]+(?:e[+-]?\d+)?)\s*$",
    re.IGNORECASE,
)

RE_START = re.compile(r"^HPL_pdgesv\(\)\s+start time\s+(.+)$")
RE_END   = re.compile(r"^HPL_pdgesv\(\)\s+end time\s+(.+)$")

# "Max aggregated wall time ..." lines
RE_WALL = re.compile(
    r"wall time\s+([a-z ]+?)\s*\.*\s*:\s*([0-9.]+)\s*$",
    re.IGNORECASE
)

# Residual pass line
RE_RESID = re.compile(
    r"^\|\|Ax-b\|\|_oo.*=\s*([0-9.eE+-]+)\s*\.+\s*(PASSED|FAILED)\s*$"
)

# Optional: parse tokens from filename if needed (fallbacks)
RE_FN_N   = re.compile(r"_N(\d+)")
RE_FN_NB  = re.compile(r"_NB(\d+)")
RE_FN_P   = re.compile(r"_P(\d+)")
RE_FN_Q   = re.compile(r"_Q(\d+)")


def parse_time_str(s):
    """Parse time strings like 'Thu Aug 14 13:52:10 2025' to ISO, else return original."""
    s = s.strip()
    for fmt in ("%a %b %d %H:%M:%S %Y", "%c"):
        try:
            return datetime.strptime(s, fmt).isoformat(sep=" ")
        except ValueError:
            pass
    return s  # keep as-is if unexpected format


def parse_single_file(path: Path):
    """Parse a single HPL .out file and return a list of row dicts."""
    rows = []
    config = {}
    start_time = None
    end_time = None
    wall_aggs = {}  # rfact/pfact/mxswp/update/laswp/up tr sv

    # Fallbacks from filename
    fn = path.name
    m = RE_FN_N.search(fn)
    if m: config.setdefault("N", int(m.group(1)))
    m = RE_FN_NB.search(fn)
    if m: config.setdefault("NB", int(m.group(1)))
    m = RE_FN_P.search(fn)
    if m: config.setdefault("P", int(m.group(1)))
    m = RE_FN_Q.search(fn)
    if m: config.setdefault("Q", int(m.group(1)))

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.rstrip("\n")

            # Global config K:V ints
            for k, rx in RE_KV_INT.items():
                m = rx.match(s)
                if m:
                    config[k] = int(m.group(1))
                    break

            # Global config K:V strs
            for k, rx in RE_KV_STR.items():
                m = rx.match(s)
                if m:
                    config[k] = m.group(1).strip()
                    break

            # SWAP
            m = RE_SWAP.match(s)
            if m:
                config["SWAP"] = m.group(1).strip()
                if m.group(2):
                    config["SWAP_threshold"] = int(m.group(2))

            # Start/end
            m = RE_START.match(s)
            if m:
                start_time = parse_time_str(m.group(1))
            m = RE_END.match(s)
            if m:
                end_time = parse_time_str(m.group(1))

            # Wall aggregates
            m = RE_WALL.search(s.lstrip("+ ").strip())
            if m:
                key = m.group(1).strip().replace(" ", "_")
                val = float(m.group(2))
                wall_aggs[key] = val

            # Residual pass/fail
            m = RE_RESID.match(s)
            if m:
                config["_residual_value"] = float(m.group(1))
                config["_residual_status"] = m.group(2).upper()

            # Results row(s)
            m = RE_RESULT_ROW.match(s)
            if m:
                variant, N, NB, P, Q, t, g = m.groups()
                row = {
                    "file": str(path),
                    "variant": variant,
                    "N": int(N),
                    "NB": int(NB),
                    "P": int(P),
                    "Q": int(Q),
                    "time_sec": float(t),
                    "gflops": float(g),
                }
                # merge global config we’ve seen so far
                for k, v in config.items():
                    if k.startswith("_"):  # internal fields handled below
                        continue
                    row[k] = v
                # start/end times & residuals if present
                if start_time: row["start_time"] = start_time
                if end_time:   row["end_time"] = end_time
                if "_residual_value" in config:
                    row["residual"] = config["_residual_value"]
                if "_residual_status" in config:
                    row["residual_status"] = config["_residual_status"]
                # wall aggs (normalize keys to a stable set)
                for k_src, v in wall_aggs.items():
                    row[f"wall_{k_src}"] = v
                rows.append(row)

    return rows


def main():
    ap = argparse.ArgumentParser(description="Parse HPL xhpl_*.out files into a CSV.")
    ap.add_argument("results_dir", type=str, help="Directory containing HPL results (*.out).")
    ap.add_argument("-o", "--output", type=str, default="hpl_results.csv", help="Output CSV path.")
    ap.add_argument("--glob", type=str, default="xhpl_*.out", help="Glob pattern to match result files.")
    args = ap.parse_args()

    root = Path(args.results_dir).expanduser().resolve()
    files = sorted(root.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched {root}/{args.glob}")

    # Collect rows
    all_rows = []
    for p in files:
        all_rows.extend(parse_single_file(p))

    if not all_rows:
        raise SystemExit("No result rows found in the provided files.")

    # Build a consistent header: preferred columns first, then any extras
    preferred = [
        "file", "variant",
        "N", "NB", "P", "Q",
        "time_sec", "gflops",
        "PMAP", "PFACT", "NBMIN", "NDIV", "RFACT", "BCAST", "DEPTH",
        "SWAP", "SWAP_threshold", "L1", "U", "EQUIL",
        "ALIGN_words",
        "start_time", "end_time",
        "residual", "residual_status",
        "wall_rfact", "wall_pfact", "wall_mxswp", "wall_update", "wall_laswp", "wall_up_tr_sv",
    ]
    # find all keys present
    all_keys = set()
    for r in all_rows:
        all_keys.update(r.keys())
    # keep preferred order, then append any remaining keys in sorted order
    header = [k for k in preferred if k in all_keys] + sorted(all_keys - set(preferred))

    # Write CSV
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"Wrote {len(all_rows)} rows to {out_path}")

if __name__ == "__main__":
    main()

