#!/usr/bin/env python3
"""
hpl_sweeper_openblas.py  (TUNING-guided variant)

Automates HPL tuning:
1) generate HPL.dat files
2) submit sbatch jobs (one per HPL.dat)
3) wait for completion, parse GFLOPS, write CSV
4) refine search around best combo with higher fidelity

Assumptions:
- A Slurm script named `xhpl_single_openblas.slurm` sits next to this file and accepts:
    sbatch xhpl_single_openblas.slurm <HPL.dat path> <tag>
  It should write result to: ${RUNROOT}/results/xhpl_<tag>.out

- RUNROOT is the directory that contains `xhpl` and the baseline HPL.dat
  Default: ~/HPL_Linpack/hpl-2.3/bin/EPYC_openblas
"""

import itertools
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- CONFIG ------------------------------------------------------------------

# HPL binary directory (where xhpl lives)
RUNROOT = Path(os.environ.get(
    "HPL_RUNROOT",
    str(Path.home() / "HPL_Linpack/hpl-2.3/bin/EPYC_openblas")
)).resolve()

# This script expects xhpl and will write results/logs here
RESULTS_DIR = RUNROOT / "results"
LOGS_DIR = RUNROOT / "logs"
SWEEPS_DIR = RUNROOT / "sweeps"
SWEEPS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Path to the Slurm batch script (single run)
SBATCH_SCRIPT = Path(__file__).parent.resolve() / "xhpl_single_openblas.slurm"

# Slurm submission behavior
SUBMIT_RATE_LIMIT = 2          # submit in bursts of this size, then wait
POLL_INTERVAL_SEC = 20         # seconds between poll checks
SACCT_AVAILABLE = True         # set False if sacct isn't available

# User-scoped squeue polling (faster on shared clusters)
SLURM_USER = os.environ.get("SLURM_USER", "kimsh")
USE_SQUEUE_USER = True  # set False to query by job IDs directly

# Total ranks (must match Slurm ntasks of your xhpl_single_openblas.slurm)
TOTAL_RANKS = 4 * 64  # 4 nodes × 64 ranks

# --------------------- TUNING-GUIDED COARSE SWEEP ----------------------------
# (Doc: NB ~ [32..256]; try Crout/Right, NDIV=2, NBMIN in {4,8}, BCAST {1,3},
#  DEPTH {0,1}. Prefer grids that keep BLACS rows on-node for 10GbE.)
def _mul_of(x, m):  # round x to nearest multiple of m (down)
    return int(x // m * m)

COARSE_N     = [ _mul_of(n, 256) for n in (147456, 196608, 262144) ]  # smaller→faster
COARSE_NB    = [128, 192, 256]
COARSE_GRIDS = [ (4,64), (8,32), (16,16) ]  # row-major mapping; 4×64 keeps rows on-node
COARSE_BCAST = [1, 3]    # 1=1rM, 3=2rM (per TUNING)
COARSE_DEPTH = [0, 1]    # per TUNING; try both
COARSE_PFACT = [1]       # Crout
COARSE_RFACT = [2]       # Right
COARSE_NBMIN = [4, 8]    # per TUNING
COARSE_NDIV  = [2]       # per TUNING
COARSE_SWAP  = [2]       # mix; we’ll set THRESH = NB automatically

# Refinement behavior
MAX_REFINEMENT_ROUNDS = 2
IMPROVE_EPS = 0.02  # 2% required improvement to keep refining

# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HplParams:
    N: int
    NB: int
    P: int
    Q: int
    PFACT: int
    RFACT: int
    NBMIN: int
    NDIV: int
    BCAST: int
    DEPTH: int
    SWAP: int
    THRESH: int  # we set this == NB by construction

    def key(self) -> str:
        # a stable filename-friendly key
        return (f"N{self.N}_NB{self.NB}_P{self.P}Q{self.Q}_pf{self.PFACT}"
                f"rf{self.RFACT}_nbmin{self.NBMIN}_ndiv{self.NDIV}"
                f"_bc{self.BCAST}_d{self.DEPTH}_sw{self.SWAP}_t{self.THRESH}")

def write_hpl_dat(path: Path, p: HplParams) -> None:
    out_abs = (RESULTS_DIR / f"xhpl_{p.key()}.out").resolve()
    txt = f"""HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
{out_abs}
8
1
{p.N}
1
{p.NB}
0
1                 # of process grids (P x Q)
{p.P}             # Ps
{p.Q}             # Qs
16.0
1                 # of panel fact
{p.PFACT}
1                 # of recursive stopping criterion
{p.NBMIN}
1                 # of panels in recursion
{p.NDIV}
1                 # of recursive panel fact.
{p.RFACT}
1                 # of broadcast
{p.BCAST}
1                 # of lookahead depth
{p.DEPTH}
{p.SWAP}          # SWAP (0=bin-exch,1=long,2=mix)
{p.THRESH}        # swapping threshold (~ NB)
0                 # L1 form (0=transposed,1=no-transposed)
0                 # U  form (0=transposed,1=no-transposed)
1                 # equilibration (0=no,1=yes)
8                 # memory alignment (double)
"""
    path.write_text(txt)

def cartesian_params(
    Ns, NBs, grids, PFACTs, RFACTs, NBMINs, NDIVs, BCASTs, DEPTHs, SWAPs
) -> List[HplParams]:
    """
    Build param combos. THRESH is always set to NB (per TUNING suggestion when SWAP=mix).
    Only keep combos where P*Q == TOTAL_RANKS.
    """
    params = []
    for (N, NB, (P,Q), PFACT, RFACT, NBMIN, NDIV, BCAST, DEPTH, SWAP) in itertools.product(
        Ns, NBs, grids, PFACTs, RFACTs, NBMINs, NDIVs, BCASTs, DEPTHs, SWAPs
    ):
        if P * Q != TOTAL_RANKS:
            continue
        params.append(HplParams(
            N=N, NB=NB, P=P, Q=Q,
            PFACT=PFACT, RFACT=RFACT, NBMIN=NBMIN, NDIV=NDIV,
            BCAST=BCAST, DEPTH=DEPTH, SWAP=SWAP, THRESH=NB
        ))
    return params

def materialize_dat_set(params: List[HplParams], out_dir: Path) -> List[Tuple[HplParams, Path]]:
    """
    Write one HPL.dat per param into out_dir. Returns list of (param, path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for p in params:
        fname = f"HPL_{p.key()}.dat"
        fpath = out_dir / fname
        write_hpl_dat(fpath, p)
        result.append((p, fpath))
    return result

def sbatch_submit(dat_file: Path, tag: str) -> Optional[str]:
    """
    Submit xhpl_single_openblas.slurm with dat_file and tag. Returns JobID or None.
    """
    cmd = ["sbatch", "--parsable", str(SBATCH_SCRIPT), str(dat_file), tag]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        # JobID may be like "66380" or "66380;cluster"
        return out.split(";")[0]
    except subprocess.CalledProcessError as e:
        print(f"[submit] failed: {e}", file=sys.stderr)
        return None

def poll_jobs(job_ids: List[str]) -> None:
    """
    Wait until all job_ids are done. Uses:
      - sacct (if available) to read terminal states
      - squeue -u <user> (preferred) or -j <ids> to see live jobs
    Prints a progress line with running/pending/done counts and elapsed time.
    """
    if not job_ids:
        return

    total = len(job_ids)
    remaining = set(job_ids)
    start = time.time()

    def _from_sacct(ids: List[str]) -> Dict[str, str]:
        if not SACCT_AVAILABLE or not ids:
            return {}
        try:
            out = subprocess.check_output(
                ["sacct", "-j", ",".join(ids), "--format=JobID,State",
                 "--parsable2", "--noheader"],
                text=True
            )
            states: Dict[str, str] = {}
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                jid, state = line.split("|", 1)
                states[jid] = state.split()[0]
            return states
        except subprocess.CalledProcessError:
            return {}

    def _from_squeue_user(ids_set: set) -> Dict[str, str]:
        try:
            if USE_SQUEUE_USER and SLURM_USER:
                out = subprocess.check_output(
                    ["squeue", "-h", "-u", SLURM_USER, "-o", "%i|%T"],
                    text=True
                )
                pairs = [ln.split("|", 1) for ln in out.strip().splitlines() if ln.strip()]
                return {jid: st for (jid, st) in pairs if jid in ids_set}
            else:
                if not ids_set:
                    return {}
                out = subprocess.check_output(
                    ["squeue", "-h", "-j", ",".join(sorted(ids_set)), "-o", "%i|%T"],
                    text=True
                )
                pairs = [ln.split("|", 1) for ln in out.strip().splitlines() if ln.strip()]
                return {jid: st for (jid, st) in pairs}
        except subprocess.CalledProcessError:
            return {}

    while remaining:
        finished = set()
        sacct_states = _from_sacct(list(remaining))
        for jid, st in sacct_states.items():
            if st and not (st.startswith("RUNNING") or st.startswith("PENDING")
                           or st.startswith("CONFIGURING") or st.startswith("COMPLETING")):
                finished.add(jid)

        squeue_states = _from_squeue_user(remaining)
        running = {jid for jid, st in squeue_states.items() if st.startswith("R")}
        pending = set(squeue_states.keys()) - running

        not_in_squeue = remaining - set(squeue_states.keys())
        finished |= not_in_squeue

        rem_after = remaining - finished
        done = total - len(rem_after)
        elapsed = int(time.time() - start)
        print(
            f"\r[wait] done={done}/{total}  running={len(running)}  "
            f"pending={len(pending)}  elapsed={elapsed//60:02d}:{elapsed%60:02d}   ",
            end="", flush=True
        )

        remaining = rem_after
        if remaining:
            time.sleep(POLL_INTERVAL_SEC)

    print("\n[wait] all jobs finished.")

def parse_gflops_from_file(out_file: Path) -> Optional[float]:
    if not out_file.exists():
        return None
    g_last = None
    wr_last = None
    try:
        with out_file.open() as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.lstrip().startswith(("WR", "WC")):
                    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?", line)
                    if nums:
                        try:
                            wr_last = float(nums[-1].replace("D", "E"))
                        except ValueError:
                            pass
                m = re.search(r"gflops\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?)",
                              line, flags=re.IGNORECASE)
                if m:
                    try:
                        g_last = float(m.group(1).replace("D", "E"))
                    except ValueError:
                        pass
        return wr_last if wr_last is not None else g_last
    except Exception:
        return None

def expected_outfile_for_tag(tag: str) -> Path:
    return RESULTS_DIR / f"xhpl_{tag}.out"

# ------------------------------ SWEEP LOGIC ----------------------------------

def coarse_round() -> List[Tuple[HplParams, float]]:
    print(f"[init] RUNROOT={RUNROOT}")
    params = cartesian_params(
        COARSE_N, COARSE_NB, COARSE_GRIDS,
        COARSE_PFACT, COARSE_RFACT, COARSE_NBMIN,
        COARSE_NDIV, COARSE_BCAST, COARSE_DEPTH,
        COARSE_SWAP
    )
    dat_dir = SWEEPS_DIR / "coarse"
    pairs = materialize_dat_set(params, dat_dir)
    print(f"[coarse] {len(pairs)} combos → {dat_dir}")

    submitted = submit_and_wait(pairs)

    results: List[Tuple[HplParams, float]] = []
    for p, _, tag in submitted:
        out = expected_outfile_for_tag(tag)
        if not out.exists() or out.stat().st_size == 0:
            print(f"[WARN] missing/empty result for {tag} → {out}", file=sys.stderr)
            continue
        gf = parse_gflops_from_file(out)
        if gf is None:
            print(f"[WARN] could not parse GFLOPS in {out}", file=sys.stderr)
            continue
        results.append((p, gf))
    csv = RESULTS_DIR / "coarse_results.csv"
    aggregate_to_csv(results, csv)
    print(f"[coarse] wrote: {csv}")
    return results

def _round_up_to_multiple(n: int, m: int) -> int:
    return int(math.ceil(n / m) * m)

def neighbor_grids(Pbest: int, Qbest: int) -> List[Tuple[int,int]]:
    """
    Keep P*Q constant; explore small perturbations near best.
    Favor shapes near {4x64, 8x32, 16x16}.
    """
    total = Pbest * Qbest
    cands = {(4, total//4) if total % 4 == 0 else (Pbest, Qbest),
             (8, total//8) if total % 8 == 0 else (Pbest, Qbest),
             (16, total//16) if total % 16 == 0 else (Pbest, Qbest)}
    # Also try divisors near sqrt(total)
    root = int(total ** 0.5)
    for delta in range(-4, 5):
        a = max(1, root + delta)
        if total % a == 0:
            b = total // a
            cands.add((a, b))
            cands.add((b, a))
    cands = {(p, q) for (p, q) in cands if p*q == total and p > 0 and q > 0}
    lst = sorted(cands)
    if (Pbest, Qbest) in lst:
        lst.remove((Pbest, Qbest))
    return [(Pbest, Qbest)] + lst[:6]  # best + up to 6 neighbors

def refine_once(prev_results: List[Tuple[HplParams, float]]) -> Tuple[List[Tuple[HplParams, float]], float]:
    """
    Build a refined grid around the best NB/PxQ/BCAST/DEPTH,
    try NB ± {32,64}, keep THRESH=NB, test the other BCAST (1<->3) and optionally 4,
    widen N modestly (rounded to multiple of 256).
    """
    best_p, best_gf = max(prev_results, key=lambda t: t[1])

    # Narrow NB band
    step = 64
    nbs = sorted(set([max(32, best_p.NB + d) for d in (-step, -32, 0, 32, step)]))
    # BCAST: keep winner + its sibling per TUNING, and also try 4 (long) once
    bcasts = sorted(set([best_p.BCAST, 3 if best_p.BCAST == 1 else 1, 4]))
    depths = sorted(set([best_p.DEPTH, 1]))  # keep best and ensure 1 is present
    grids = neighbor_grids(best_p.P, best_p.Q)[:4]  # best + a few neighbors
    nbmin = sorted(set([best_p.NBMIN, 4, 8]))
    ndiv  = [2]
    pfact = [1]
    rfact = [2]

    # Increase N a bit; keep multiples of 256
    Ns_raw = [best_p.N, int(best_p.N*1.33), int(best_p.N*1.5)]
    Ns = sorted({_round_up_to_multiple(n, 256) for n in Ns_raw})

    swap = [2]  # mix

    refined: List[HplParams] = []
    for (N, NB, (P, Q), BCAST, DEPTH, NBMIN, NDIV) in itertools.product(
        Ns, nbs, grids, bcasts, depths, nbmin, ndiv
    ):
        if P * Q != TOTAL_RANKS:
            continue
        refined.append(HplParams(
            N=N, NB=NB, P=P, Q=Q, PFACT=1, RFACT=2,
            NBMIN=NBMIN, NDIV=NDIV, BCAST=BCAST, DEPTH=DEPTH,
            SWAP=2, THRESH=NB
        ))

    dat_dir = SWEEPS_DIR / f"refine_{int(time.time())}"
    pairs = materialize_dat_set(refined, dat_dir)
    print(f"[refine] {len(pairs)} combos → {dat_dir}")

    submitted = submit_and_wait(pairs)

    results: List[Tuple[HplParams, float]] = []
    for p, _, tag in submitted:
        out = expected_outfile_for_tag(tag)
        gf = parse_gflops_from_file(out)
        if gf is not None:
            results.append((p, gf))
        else:
            print(f"[WARN] no GFLOPS parsed for {tag}", file=sys.stderr)

    csv = RESULTS_DIR / "refine_results.csv"
    aggregate_to_csv(results, csv)
    print(f"[refine] appended: {csv}")

    new_best = max(results, key=lambda t: t[1])[1] if results else -1.0
    return results, new_best / best_gf if best_gf > 0 else 0.0

# ------------------------------- I/O UTILS -----------------------------------

def aggregate_to_csv(rows: List[Tuple[HplParams, float]], csv_path: Path) -> None:
    header = ("N,NB,P,Q,PFACT,RFACT,NBMIN,NDIV,BCAST,DEPTH,SWAP,THRESH,GFLOPS\n")
    new_file = not csv_path.exists()
    with csv_path.open("a") as f:
        if new_file:
            f.write(header)
        for p, gf in rows:
            f.write(f"{p.N},{p.NB},{p.P},{p.Q},{p.PFACT},{p.RFACT},"
                    f"{p.NBMIN},{p.NDIV},{p.BCAST},{p.DEPTH},{p.SWAP},{p.THRESH},{gf:.6g}\n")

# ------------------------------ SUBMIT + WAIT --------------------------------

def submit_and_wait(dat_paths: List[Tuple[HplParams, Path]]) -> List[Tuple[HplParams, str, str]]:
    """
    Submit all jobs. Returns list of (params, jobid, tag).
    Submits in bursts and waits between bursts to avoid flooding the scheduler.
    """
    submitted: List[Tuple[HplParams, str, str]] = []
    burst: List[Tuple[HplParams, str, str]] = []

    total = len(dat_paths)
    print(f"[submit] total jobs: {total}")
    for idx, (p, fpath) in enumerate(dat_paths, 1):
        tag = p.key()
        jid = sbatch_submit(fpath, tag)
        if jid:
            submitted.append((p, jid, tag))
            burst.append((p, jid, tag))
            print(f"[submit] {idx}/{total}  job={jid}  tag={tag}")
        else:
            print(f"[WARN] failed to submit: {fpath.name}", file=sys.stderr)

        # throttle submission bursts
        if len(burst) >= SUBMIT_RATE_LIMIT:
            poll_jobs([jid for _, jid, _ in burst])
            burst.clear()

    # wait for the rest
    if burst:
        poll_jobs([jid for _, jid, _ in burst])

    # final safety: ensure all are done
    poll_jobs([jid for _, jid, _ in submitted])
    return submitted

# ---------------------------------- MAIN -------------------------------------

def main():
    print(f"[init] RUNROOT={RUNROOT}")
    if not (RUNROOT / "xhpl").exists():
        print("ERROR: xhpl not found under RUNROOT. Check RUNROOT or build HPL.", file=sys.stderr)
        sys.exit(2)
    if not SBATCH_SCRIPT.exists():
        print("ERROR: xhpl_single_openblas.slurm not found next to this script.", file=sys.stderr)
        sys.exit(2)

    # 1) coarse
    coarse = coarse_round()
    if not coarse:
        print("No coarse results collected.", file=sys.stderr)
        sys.exit(1)

    # 2) refine rounds
    cur = coarse
    for r in range(1, MAX_REFINEMENT_ROUNDS + 1):
        print(f"[refine] round {r} …")
        refined, ratio = refine_once(cur)
        if not refined:
            print("[refine] got no results; stopping.")
            break
        print(f"[refine] best improvement ratio={ratio:.3f}")
        if ratio < (1.0 + IMPROVE_EPS):
            print("[refine] improvement below threshold; stopping.")
            break
        cur = cur + refined

    # Final best
    all_res = cur
    best_p, best_gf = max(all_res, key=lambda t: t[1])
    print("\n=== BEST FOUND ===")
    for k, v in asdict(best_p).items():
        print(f"{k:7s} = {v}")
    print(f"GFLOPS  = {best_gf:.6g}")

if __name__ == "__main__":
    main()

