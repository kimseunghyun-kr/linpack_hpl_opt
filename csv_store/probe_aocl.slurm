#!/bin/bash
#SBATCH -J blas_probe                 # job name
#SBATCH -N 1 --constraint=xcnf        # run on one EPYC node
#SBATCH -t 00:02:00                   # 2-minute wall clock
#SBATCH -o blas_probe_%j.out          # write output here

echo "===========  BLAS / LAPACK PROBE  ==========="

###############################################################################
#  1) Environment-Modules?  (shows what the admins expose)                    #
###############################################################################
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

if type module &>/dev/null; then
    echo -e "\n== module avail (keywords: aocl, mkl, openblas, blis, essl) =="
    # -t prints terse list; grep filters by common BLAS names
    module -t avail 2>&1 | grep -Ei '(aocl|mkl|openblas|blis|essl|atlas)' || \
        echo "(no BLAS-related modules found)"
else
    echo -e "\n(module command not present)"
fi

###############################################################################
#  2) ldconfig cache — shared objects already on the node                     #
###############################################################################
echo -e "\n== ldconfig -p (searching for BLAS libraries) =="
ldconfig -p | grep -Ei '(openblas|blis|mkl|atlas|essl)' || \
    echo "(nothing in ldconfig cache)"

###############################################################################
#  3) Static archives or uncommon paths — fast filesystem scan                #
###############################################################################
echo -e "\n== filesystem scan (up to depth 4) =="
for prefix in /opt /usr /usr/local; do
  [ -d "$prefix" ] && \
  find "$prefix" -maxdepth 4 -type f \( \
        -name 'libopenblas*.a' -o \
        -name 'libopenblas*.so' -o \
        -name 'libblis*.a'     -o \
        -name 'libblis*.so'    -o \
        -name 'libmkl_intel_lp64*.a' -o \
        -name 'libmkl_intel_lp64*.so' \) -print 2>/dev/null | head -n 10
done | sed 's/^/  /' || echo "(no static archives located)"

###############################################################################
#  4) Quick runtime check — does OpenBLAS export its config string?           #
###############################################################################
echo -e "\n== OpenBLAS runtime self-test (if present) =="
cat > check_openblas.c <<'C_SRC'
#include <stdio.h>
#ifdef __cplusplus
extern "C"
#endif
const char* openblas_get_config(void);
int main(){ if(openblas_get_config) printf("%s\n",openblas_get_config()); return 0; }
C_SRC

mpicc -O2 -o check_openblas check_openblas.c 2>/dev/null && \
./check_openblas 2>/dev/null || echo "(openblas_get_config symbol not found)"

echo "===========  END PROBE  ==========="

