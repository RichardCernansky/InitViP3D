#!/usr/bin/env bash
# Find the largest samples_per_gpu that fits, by training each size for a few
# seconds and reading the peak memory mmdet logs.
#
#   bash tools/probe_batch_size.sh CONFIG [SIZES...]
#
# Env:
#   SECS=90                      seconds of training per size
#   WORK_DIR=work_dirs/perun/bs_probe
#   EXTRA="load_from=... k=v"    extra --cfg-options entries
#
# Reports peak MiB per size and stops at the first out-of-memory.
set -u

CONFIG=${1:?usage: probe_batch_size.sh CONFIG [SIZES...]}
shift
SIZES=${*:-1 2 4 6 8 10 12}
SECS=${SECS:-90}
WORK_DIR=${WORK_DIR:-work_dirs/perun/bs_probe}
EXTRA=${EXTRA:-}

total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "config : $CONFIG"
echo "gpu    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), ${total} MiB"
echo "budget : ${SECS}s per size"
echo
printf '%-6s %-12s %-10s %s\n' 'batch' 'peak MiB' '% of GPU' 'result'

for B in $SIZES; do
    out=$(timeout "$SECS" python tools/train.py "$CONFIG" \
            --work-dir "$WORK_DIR" \
            --cfg-options data.samples_per_gpu="$B" \
                          log_config.interval=1 \
                          evaluation.interval=99999 \
                          checkpoint_config.interval=99999 \
                          model.bev_vis=False \
                          $EXTRA 2>&1)

    iters=$(grep -c 'Epoch \[1\]\[' <<<"$out")
    peak=$(grep -o 'memory: [0-9]*' <<<"$out" | awk '{print $2}' | sort -n | tail -1)

    if grep -q 'out of memory' <<<"$out"; then
        printf '%-6s %-12s %-10s %s\n' "$B" "${peak:--}" '-' 'OOM -- stopping'
        break
    elif [ "$iters" -eq 0 ]; then
        printf '%-6s %-12s %-10s %s\n' "$B" '-' '-' 'no iteration finished (see below)'
        grep -iE 'error|Traceback' <<<"$out" | head -3 | sed 's/^/       /'
        break
    else
        printf '%-6s %-12s %-10s %s\n' \
            "$B" "$peak" "$(awk -v p="$peak" -v t="$total" 'BEGIN{printf "%.0f%%", 100*p/t}')" \
            "ok (${iters} iters)"
    fi
done

echo
echo "Note: 'peak MiB' is torch's allocated peak, which runs below what nvidia-smi"
echo "shows. Leave headroom -- a hard scene can exceed a short probe's peak."
