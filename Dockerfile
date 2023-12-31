FROM runpod/base:0.4.0-cuda11.8.0

# System dependencies
COPY builder/setup.sh /setup.sh
RUN /bin/bash /setup.sh && \
    rm /setup.sh

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

ADD src .

COPY TinyStories-ihvp-1M.pt /TinyStories-ihvp-1M.pt

CMD ["python", '-u', "/handler.py"]
