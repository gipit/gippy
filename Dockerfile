FROM developmentseed/geolambda:latest
 
WORKDIR /build

COPY requirements*txt /build/
RUN \
    pip install -r requirements.txt; \
    pip install -r requirements-dev.txt;

COPY . /build
RUN \
    pip install . -v; \
    rm -rf /build/*;

WORKDIR /home/geolambda
