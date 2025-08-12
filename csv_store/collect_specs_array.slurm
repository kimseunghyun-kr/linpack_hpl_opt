#!/bin/bash
#SBATCH --job-name=specs_array
#SBATCH --output=collect_specs_array_%A_%a.out
#SBATCH --error=collect_specs_array_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --partition=normal
#SBATCH --array=0-5

# Map task ID to actual physical node name
NODES=(xcnc0 xcnd0 xcne0 xcnf0 xcng0 xcnz0)
TARGET_NODE=${NODES[$SLURM_ARRAY_TASK_ID]}

# Only run if this task is allocated to the correct node
if [[ "$(hostname)" != "$TARGET_NODE" ]]; then
    echo "Skipping: running on $(hostname), but assigned to $TARGET_NODE"
    exit 0
fi

OUTFILE="$SLURM_SUBMIT_DIR/spec_${TARGET_NODE}.csv"

CPU=$(lscpu)
MEM=$(free -h | grep Mem | awk '{print $2}')
HOST=$(hostname)

echo "Node,Model,CPU(s),Cores/Socket,Threads/Core,Sockets,Memory(GB)" > "$OUTFILE"
echo "$HOST,$(echo "$CPU" | grep "Model name:" | sed "s/Model name:[ \t]*//" | sed "s/,//g"),\
$(echo "$CPU" | grep "^CPU(s):" | awk '{print $2}'),\
$(echo "$CPU" | grep "Core(s) per socket:" | awk '{print $4}'),\
$(echo "$CPU" | grep "Thread(s) per core:" | awk '{print $4}'),\
$(echo "$CPU" | grep "Socket(s):" | awk '{print $2}'),\
$MEM" >> "$OUTFILE"

