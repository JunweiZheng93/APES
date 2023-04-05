CONFIG=$1
CHECKPOINT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher none
