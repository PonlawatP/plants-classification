# syntax=docker/dockerfile:1
FROM python:3.10.12-slim-bullseye
WORKDIR /Plants
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY ./app ./app
COPY ./model ./model
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]