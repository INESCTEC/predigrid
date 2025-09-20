# Use an official Python runtime as a parent image
FROM python:3.8-slim

RUN apt-get update && apt-get -y install nano htop vim tree
RUN pip install --upgrade pip

# Set the working directory to /app
WORKDIR /flask_app

# Copying the requirements for installation to take
# advantage of the caching.
COPY /flask_app/requirements.txt .
RUN pip install -r ./requirements.txt

# Copy the current directory contents into the container at /app
COPY /flask_app .

# Copy Gunicorn configuration file:
COPY /flask_app/gunicorn.conf .

# api -> name of the running script (without .py) / app -> name of flask_app to run:
CMD gunicorn -c gunicorn.conf app:app -b :5001 --name=app --timeout=60 --log-level=debug --log-file=-
#CMD ["tail", "-f", "/dev/null"]
