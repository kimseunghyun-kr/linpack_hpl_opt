#!/usr/bin/env bash
set -euo pipefail

# Use your user-space MPICH first
export PATH="$HOME/.local/mpich/bin:$PATH"

# One thread per rank
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Optional: routable iface hint (e.g., eth0|eno1|ib0|enp131s0f0)
IFACE="${IFACE:-}"

# Build the probe binary (C++ source)
SRC="${SRC:-provider_probe.cpp}"
BIN="$HOME/provider_probe_bin"    
mpicxx -O2 -std=c++11 -o "$BIN" "$SRC"

# Candidate provider settings (name → env)
declare -A PROV_ENV
PROV_ENV[cxi]="FI_PROVIDER=cxi"
PROV_ENV[rxm+verbs]="FI_PROVIDER=rxm,verbs FI_OFI_RXM_USE_SRX=1"
PROV_ENV[verbs]="FI_PROVIDER=verbs"
PROV_ENV[psm3]="FI_PROVIDER=psm3"
PROV_ENV[tcp]="FI_PROVIDER=tcp"

# Apply iface hints where relevant
if [[ -n "$IFACE" ]]; then
  PROV_ENV[verbs]="${PROV_ENV[verbs]} FI_VERBS_IFACE=$IFACE"
  PROV_ENV[rxm+verbs]="${PROV_ENV[rxm+verbs]} FI_VERBS_IFACE=$IFACE"
  PROV_ENV[tcp]="${PROV_ENV[tcp]} FI_TCP_IFACE=$IFACE"
fi

echo "=== Node list sanity (need 2 nodes in allocation) ==="
srun -N2 -n2 --mpi=pmi2 hostname | sort -u

echo -e "\n=== fi_info provider inventory (1 node) ==="
if command -v fi_info >/dev/null 2>&1; then
  srun -N1 -n1 --mpi=pmi2 bash -lc 'fi_info -p | awk "/provider:/ {print \$2}" | sort -u'
else
  echo "fi_info not found; skipping provider inventory."
fi

echo -e "\n=== Provider connectivity probes (2 ranks across 2 nodes) ==="
for key in cxi rxm+verbs verbs psm3 tcp; do
  echo -e "\n--- Trying provider: $key ---"
  ENVSTR=${PROV_ENV[$key]}
  echo "ENV: $ENVSTR"
  # Run 2 ranks (1 per node), show binding for transparency
  if srun -N2 -n2 --mpi=pmi2 --cpu-bind=cores env $ENVSTR "$BIN" 2>&1 | tee "/tmp/probe_${key}.log"; then
    echo "[$key] ✅ PASS"
  else
    echo "[$key] ❌ FAIL"
    case "$key" in
      psm3)
        echo "   Hint: PSM3 targets Intel Omni-Path; subnet/VLAN mismatches or missing OPA will time out."
        ;;
      verbs|rxm+verbs)
        echo "   Hint: Requires InfiniBand (mlx5_*). Try IFACE=ib0 (export IFACE=ib0) and rerun."
        ;;
      cxi)
        echo "   Hint: Only valid on HPE Slingshot/CXI systems; absence is expected to fail."
        ;;
      tcp)
        echo "   Hint: Should work if IP routing exists. Try IFACE=eth0 if multiple NICs are present."
        ;;
    esac
  fi
done

echo -e "\nDone. Logs written to /tmp/probe_*.log"

