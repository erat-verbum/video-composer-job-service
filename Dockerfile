# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Copy test data (expected to be provided via volume mount at /app/data)
# The data directory is mounted via docker-compose

# Set PYTHONPATH before installing/running
ENV PYTHONPATH=/app

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y make libatomic1 && \
    make install && \
    make lint && \
    make check && \
    make test-unit

# Create non-root user with UID 1000 and GID 1000
RUN groupadd -r -g 1000 appgroup && useradd -r -u 1000 -g appgroup appuser

RUN mkdir -p /home/appuser/.cache && chown appuser:appgroup /home/appuser/.cache

# Set ownership of application directory
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER 1000:1000

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Run app via the template Makefile target (which itself uses uv)
CMD ["make", "run"]
