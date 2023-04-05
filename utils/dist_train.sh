CONFIG=$1
GPUS=$2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch
