# Use the specified Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONUNBUFFERED=TRUE
ENV RUNNING_IN_DOCKER=true

# Update the system and install necessary packages
RUN apt-get update && \
    apt-get install -y sudo build-essential python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the Docker container
WORKDIR /src

# Copy the requirements.txt file and install Python dependencies
COPY ./requirements.txt /src/requirements.txt
RUN pip install --no-cache-dir -r /src/requirements.txt

# Copy the necessary folders and files
COPY ./src /src
COPY ./notebooks/eda.ipynb /src/notebooks/

# Add a new user and switch to it
RUN adduser --disabled-password --gecos '' user
USER user
