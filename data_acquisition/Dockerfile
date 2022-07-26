# syntax=docker/dockerfile:1

FROM eldritchjs/data_acquisition:1.0.0

USER 0
COPY . /opt/app-root
WORKDIR /opt/app-root
RUN pip3 install -r requirements.txt
CMD python3 app.py

