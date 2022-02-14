FROM nvcr.io/nvidia/pytorch:21.05-py3
COPY requirements.txt /tmp/requirements.txt
COPY cache.py /tmp/cache.py
RUN pip install -r /tmp/requirements.txt
RUN python /tmp/cache.py
