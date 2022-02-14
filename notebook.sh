docker run --gpus all --rm \
    -v `pwd`:/work \
    -w /work \
    -p 8888:8888 \
    -it \
    transformers \
    jupyter-lab --allow-root --ip=0.0.0.0 --no-browser
