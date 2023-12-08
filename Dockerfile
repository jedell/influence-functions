# Python 3.11 runtime image
FROM python:3.11-slim-bookworm

# setup python env
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install gdown
# RUN gdown
# RUN gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx -O openwebtext.tar.xz
# RUN mkdir /openwebtext
# RUN tar -xf openwebtext.tar.xz -C /openwebtext
# RUN rm openwebtext.tar.xz

COPY model.py .
COPY dataset.py .
COPY ihvp.py .

CMD python ihvp.py --path /openwebtext


