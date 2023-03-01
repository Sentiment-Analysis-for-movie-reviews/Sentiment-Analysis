#!/bin/bash

FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04


RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install in batch (silent) mode, does not edit PATH or .bashrc or .bash_profile
# -p path
# -f force
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

ENV PATH=/root/miniconda3/bin:${PATH} 

#RUN source /root/.bashrc
#RUN source /root/.bash_profile

RUN conda update -y conda
RUN conda list
RUN conda install -y numpy \
    matplotlib \
    pandas
RUN rm Miniconda3-latest-Linux-x86_64.sh

WORKDIR /app


COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["python", "main.py"]