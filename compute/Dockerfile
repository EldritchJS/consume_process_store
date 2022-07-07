# syntax=docker/dockerfile:1

FROM registry.access.redhat.com/ubi8/python-39

USER 0
RUN yum -y install python39 python39-devel mesa-libGL postgresql-devel gcc make
COPY . /opt/app-root
WORKDIR /opt/app-root
RUN pip3 install -r ./requirements.txt

CMD python3 app.py

