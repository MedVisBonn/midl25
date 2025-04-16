#!/usr/bin/env bash

docker run \
	-it \
	--net=host \
	--runtime=nvidia \
	--ipc=host \
	--gpus all \
	--cpus="16" \
	--privileged \
	--mount type=bind,source="/path/to/data",target=/root/data \
	--mount type=bind,source="/path/to/out",target=/root/out \
	midl25-190/test:1.0
