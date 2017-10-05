FROM ubuntu:latest
MAINTAINER Ben Johnston <ben.johnston@sydney.edu.au>

LABEL Description="" Vendor="University of Sydney" Version="0.1"

# Replace mirror for quicker repo access
RUN sed -i "s/archive.ubuntu.com/au.archive.ubuntu.com/g" /etc/apt/sources.list

# Install depdendencies
RUN apt-get -y update
RUN apt-get install -y python3 python3-pip python3-dev wget python3-tk x11vnc xvfb
RUN apt-get install -y libboost-python-dev build-essential cmake libgtk2.0-dev
RUN apt-get install -y python3-numpy git

# Localisation configuration
#RUN locale-gen en_US.UTF-8
#ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN mkdir -p /etc/sudoers.d/
RUN touch /etc/sudoers.d/doc-E-brown
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/doc-E-brown && \
    echo "doc-E-brown:x:${uid}:${gid}:doc-E-brown,,,:/home/doc-E-brown:/bin/bash" >> /etc/passwd && \
    echo "doc-E-brown:x:${uid}:" >> /etc/group && \
    echo "doc-E-brown ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/doc-E-brown && \
    chmod 0440 /etc/sudoers.d/doc-E-brown && \
    chown ${uid}:${gid} -R /home/doc-E-brown

# Install python dependencies
RUN wget https://github.com/opencv/opencv/archive/3.3.0.tar.gz
RUN tar -xzvf 3.3.0.tar.gz 
RUN mkdir -p /opencv-3.3.0/build
RUN cd /opencv-3.3.0/build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3.5/dist-packages/numpy/core/include/ \
    -D PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.so ..
RUN cd /opencv-3.3.0/build && make -j 4
RUN rm 3.3.0.tar.gz

# Install the source code
RUN pip3 install virtualenv virtualenvwrapper 
RUN apt-get install -y vim

USER doc-E-brown

ENV WORKON_HOME=/home/doc-E-brown/envs
ENV VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
RUN mkdir -p $WORKON_HOME
RUN echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
RUN cd /home/doc-E-brown && \ 
    git clone https://github.com/doc-E-brown/FacialLandmarkingReview.git

# Create the virtualenvs 
RUN cd /home/doc-E-brown/FacialLandmarkingReview/experiments && \
    bash -c "source /usr/local/bin/virtualenvwrapper.sh && mkvirtualenv Sec3_FeatureExtraction" && \
    bash -c "source /usr/local/bin/virtualenvwrapper.sh && mkvirtualenv -p $(which python) Sec4_FeatureExtraction" && \
    bash -c "source /usr/local/bin/virtualenvwrapper.sh && workon Sec3_FeatureExtraction && pip install dlib scipy pillow numpy" && \
    bash -c "source /usr/local/bin/virtualenvwrapper.sh && workon Sec3_FeatureExtraction && add2virtualenv /opencv-3.3.0/build/lib/python3"

RUN mkdir -p /home/doc-E-brown/datasets
ENV DATASETS=/home/doc-E-brown/datasets
ENV BIOID_DATA=$DATASETS/BioID
ENV MUCT_DATA=$DATASETS/MUCT
ENV MUCT_DATA=$DATASETS/HELEN

ENV HOME /home/doc-E-brown
