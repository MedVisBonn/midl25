#!/usr/bin/env bash

docker buildx build \
	--pull \
	--progress=plain \
	--ssh default \
	-t midl25-190/test:1.0 \
	-f Dockerfile .
