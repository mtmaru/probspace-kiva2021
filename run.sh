docker run --gpus all --rm \
    -v `pwd`:/work \
    -w /work \
    -it \
    transformers \
    python $1
