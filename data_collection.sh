#!/bin/bash
#Usage: please specify oneshot dataset directory to train and test, and directory name to save the collection data
#like: ./data_collection.sh ./table/table_9 progress

set -x
set -e
if [ $# -eq 0 ]
  then
    python main.py --train False
else
	dataset=$1
	saved_name=$2
	python main.py --train False --train_test_dataset ${dataset} --dataset ${saved_name}
fi

set +x