# Use the specified Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV RUNNING_IN_DOCKER=true

# Update the system and install necessary packages
RUN apt-get update && \
    apt-get install -y sudo build-essential python3-pip python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the Docker container
WORKDIR /src

# Copy pipfile requirements
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install project dependencies
RUN pipenv install --deploy --system && \
rm -rf /root/.cache

# Copy your source code and notebooks
# Copy your source code, notebooks, models, and data
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY models/ ./models/
COPY data/ ./data/

# Add a new user and switch to it
RUN adduser --disabled-password --gecos '' user
USER user
