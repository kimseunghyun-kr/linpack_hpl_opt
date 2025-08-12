#!/bin/bash
#SBATCH -J aocl_deep_probe
#SBATCH -N 1 --constraint=xcnf
#SBATCH -t 00:05:00
#SBATCH -o aocl_deep_%j.out

echo "==== deep AOCL / BLIS scan on $(hostname) ===="

# 1) list candidate roots: /opt /usr /apps /software /share /cm ...
for root in $(find / -maxdepth 1 -type d \( -perm -o+x \) 2>/dev/null | grep -Ev '^/(proc|sys|dev|run|tmp|home)'); do
  echo "-- scanning $root"
  find "$root" -maxdepth 8 \
       \( -iname 'libblis*.a' -o -iname 'libblis*.so' \
          -o -iname 'libflame*.a' -o -iname 'libflame*.so' \
          -o -type d -iname '*aocl*' \) 2>/dev/null | head -n 20
done

