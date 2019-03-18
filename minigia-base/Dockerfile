FROM fedora:29

RUN dnf -y update && \
    dnf clean all

RUN dnf -y install gcc-c++ git cmake
RUN dnf -y install mpich-devel make
RUN dnf -y install boost boost-devel
RUN dnf -y install boost-python3
RUN dnf -y install fftw3-mpich-devel
RUN dnf -y install google-benchmark-devel
