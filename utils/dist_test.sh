CONFIG=$1
CHECKPOINT=$2
GPUS=$3
VIS=$4

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    $VIS
