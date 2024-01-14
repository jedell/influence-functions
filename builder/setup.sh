#!/bin/bash

# Install AWS CLI
apt-get update && apt-get install -y \
    python3-pip \
    groff-base \
    && rm -rf /var/lib/apt/lists/*

pip3 install --upgrade awscli

# Set AWS credentials
export AWS_ACCESS_KEY_ID=$1
export AWS_SECRET_ACCESS_KEY=$2
export AWS_DEFAULT_REGION=us-east-2

# Pull file from S3
export AWS_S3_BUCKET=$3
aws s3 cp s3://$AWS_S3_BUCKET/TinyStories-ihvp-f.pt /TinyStories-ihvp-f.pt

