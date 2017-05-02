#!/bin/bash
#Usage: please specify oneshot dataset directory to train and test, and training image basename
#like: ./oneshot_training.sh ./table/table_9 "001"
set -x
set -e
if [ $# -eq 0 ]
  then
    python main.py
else
	dataset=$1
	image=$2
	python main.py --train_test_dataset ${dataset} --oneshot_img ${image}
fi

set +x