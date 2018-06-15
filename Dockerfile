FROM developmentseed/geolambda:latest
 
WORKDIR /build

RUN \
    yum install -y swig;

COPY requirements*txt /build/
RUN \
    pip2 install -r requirements.txt; \
    pip2 install -r requirements-dev.txt; \
    pip3 install -r requirements.txt; \
    pip3 install -r requirements-dev.txt;

COPY . /build
RUN \
    git clean -xfd; \
    pip2 install .; \
    git clean -xfd; \
    pip3 install .; \
    rm -rf /build/*;

WORKDIR /home/geolambda
