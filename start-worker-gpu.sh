#!/bin/bash

SYSTEM_PATH=`dirname "$0"`

export LD_LIBRARY_PATH=$SYSTEM_PATH/lib
export PYTHONPATH=$SYSTEM_PATH/pylib

GPU="$CUDA_VISIBLE_DEVICES"
SERVER="0.0.0.0"
PORT="${3:-60019}"

pythonCMD="python -u -W ignore"

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$GPU $pythonCMD worker.py \
	--server ${SERVER} \
	--port ${PORT} \
	--name "asr-cs" \
	--fingerprint "cs-CS" \
	--outfingerprint "cs-CS" \
	--inputType "audio" \
	`#--outputType "unseg-text"` \
	--outputType "text" \
	--dict "../model_1024/m4k.dict" \
	--model "../model_1024/epoch-avg.dic" \
	--beam-size 8 --fp16 

