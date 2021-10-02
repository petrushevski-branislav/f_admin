# get slim python as base image
FROM python:3.8-slim

# workdir
WORKDIR /src

# copy files into workdir
COPY . .

# install needed packages
RUN pip3 install pika

# run main.py script
CMD ["python", "-u", "./main.py"]