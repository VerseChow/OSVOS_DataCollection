#!/bin/bash
#Usage: please specify oneshot dataset directory to train and test, and directory name to save the collection data
#also you could then specify name of jpg and xml to save and label to save
#like: ./data_collection.sh ./table/table_9 progress table_9_ table

set -x
set -e
if [ $# -eq 0 ]
  then
    python main.py --train False
else
	dataset=$1
	saved_name=$2
	jpg_xml_name=$3
	label=$4
	python main.py --train False --train_test_dataset ${dataset} --dataset ${saved_name}\
	--saved_name ${jpg_xml_name} --label ${label}
fi

set +x