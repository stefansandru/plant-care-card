FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Set platform for Apple Silicon compatibility
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl python3 python3-pip

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

RUN pip3 --timeout=300 --no-cache-dir install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

COPY ./requirements.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt

# Copy model files
COPY ./efficint_net_b1_model /efficint_net_b1_model

# Copy app files
COPY ./app /app
WORKDIR /app/
ENV PYTHONPATH=/app
RUN ls -lah /app/*

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 80
CMD ["/start.sh"]
