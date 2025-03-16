FROM python:3.10.16-slim

RUN pip install obp==0.5.7 matplotlib==3.7.3 japanize_matplotlib==1.1.3

WORKDIR /work
