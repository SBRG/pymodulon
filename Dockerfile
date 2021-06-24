ARG BASE_CONTAINER=jupyter/minimal-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Anand Sastry <avsastry@eng.ucsd.edu>"

USER root

# Install GraphViz, ncbi-blast, cm-super+dvipng for latex labels,
# cpanminus+libexpat1+zlib1g-dev for MEME
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends dvipng cm-super graphviz ncbi-blast+ \
        cpanminus libexpat1-dev zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install MEME pre-requisites
RUN cpanm File::Which HTML::PullParser HTML::Template \
          HTML::TreeBuilder JSON XML::Simple XML::Parser::Expat

USER $NB_UID

# Download MEME
RUN wget -q https://meme-suite.org/meme/meme-software/5.3.3/meme-5.3.3.tar.gz && \
    tar xzf meme-5.3.3.tar.gz && \
    rm -r meme-5.3.3.tar.gz

# Install MEME
RUN cd meme-5.3.3 && \
    ./configure --prefix=$HOME/meme --enable-build-libxml2 --enable-build-libxslt && \
    make && \
    make install && \
    cd .. && \
    rm -r meme-5.3.3

# Update PATHs
ENV PATH="${HOME}/meme/bin:${HOME}/meme/libexec/meme-5.3.3:${PATH}"

# Define pymodulon version
ARG PYMODULON_VERSION=0.2.1

# Install pymodulon and requirements
RUN echo $PYMODULON_VERSION
RUN pip install pymodulon==$PYMODULON_VERSION
