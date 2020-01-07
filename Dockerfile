FROM developmentseed/geolambda:latest
 
WORKDIR /build

RUN \
    yum install -y swig \
        && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
        && python3 get-pip.py

COPY requirements*txt /build/
RUN \
    pip3 install -r requirements.txt; \
    pip3 install -r requirements-dev.txt;

COPY . /build
RUN \
    git clean -xfd; \
    pip3 install . ; \
    rm -rf /build/*;

WORKDIR /home/geolambda
