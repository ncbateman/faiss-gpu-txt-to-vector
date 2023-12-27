# Use the official Ubuntu image as a parent image
FROM ragbase:latest

RUN pip3 install --default-timeout=1000 --ignore-installed pytest
RUN pip3 install --default-timeout=1000 --ignore-installed pytest-mock

# Set working directory
WORKDIR /app

RUN mkdir -p models

# Copy the necessary files from your host to the container
COPY . .

# Command to run when the container starts
CMD ["pytest"]
