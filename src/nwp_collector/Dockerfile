# Use an official Python runtime as a parent image
FROM python:3.8

MAINTAINER joao.p.viana@inesctec.pt

# Set the working directory to /app
WORKDIR /app

#Install Cron:
RUN apt-get update && apt-get -y install nano htop

# Copy requirements:
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]