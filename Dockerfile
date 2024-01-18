FROM runpod/base:0.4.0-cuda11.8.0

# System dependencies
COPY builder/setup.sh /setup.sh
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG S3_URI
RUN /bin/bash /setup.sh ${AWS_ACCESS_KEY_ID} ${AWS_SECRET_ACCESS_KEY} ${S3_URI}
RUN rm /setup.sh

# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

ADD src .

CMD ["python", '-u', "/handler.py"]
