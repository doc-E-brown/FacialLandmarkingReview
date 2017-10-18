FROM continuumio/anaconda:latest
MAINTAINER Ben Johnston <ben.johnston@sydney.edu.au>

LABEL Description="Facial Landmarking Review Paper Sec4" Vendor="University of Sydney" Version="0.1"

RUN mkdir -p /etc/sudoers.d/
RUN touch /etc/sudoers.d/doc-E-brown
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/doc-E-brown && \
    echo "doc-E-brown:x:${uid}:${gid}:doc-E-brown,,,:/home/doc-E-brown:/bin/bash" >> /etc/passwd && \
    echo "doc-E-brown:x:${uid}:" >> /etc/group && \
    echo "doc-E-brown ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/doc-E-brown && \
    chmod 0440 /etc/sudoers.d/doc-E-brown && \
    chown ${uid}:${gid} -R /home/doc-E-brown

RUN conda install -c menpo menpoproject --yes


USER doc-E-brown
RUN cd /home/doc-E-brown && \ 
    git clone https://github.com/doc-E-brown/FacialLandmarkingReview.git

RUN conda create -n FacialLandmarkingReview
RUN bash -c "source activate FacialLandmarkingReview && conda install -c menpo menpofit --yes"

RUN mkdir -p /home/doc-E-brown/datasets
ENV DATASETS /home/doc-E-brown/datasets
ENV BIOID_DATA $DATASETS/BioID
ENV MUCT_DATA $DATASETS/MUCT
ENV HELEN_DATA $DATASETS/HELEN
ENV IBUG_DATA $DATASETS/ibug/300W/
ENV MENPO_DATA $DATASETS/ibug/menpo_2017_trainset

ENV HOME /home/doc-E-brown
RUN echo 'cd $HOME' >> $HOME/.bashrc
RUN echo 'source activate FacialLandmarkingReview' >> $HOME/.bashrc
ENTRYPOINT bash
