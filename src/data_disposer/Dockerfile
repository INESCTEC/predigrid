# Use an official Python runtime as a parent image
FROM python:3.8

MAINTAINER jose.r.andrade@inesctec.pt

RUN apt-get update && apt-get -y install cron nano htop vim tree dos2unix

# Magic Intro:
WORKDIR /root
COPY /system/custom_bashrc .bashrc
COPY /system/intro.txt intro.txt
RUN dos2unix .bashrc

# Set the working directory to /app
WORKDIR /app

# Copying the requirements for installation to take
# advantage of the caching.
COPY requirements.txt .
RUN pip install -r ./requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Create shell script to execute disposer pipeline
RUN chmod +x run_disposer
RUN cp run_disposer /usr/bin

CMD run_disposer