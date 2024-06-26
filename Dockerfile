# Use an official Python runtime as a parent image
FROM python:3.10.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app.py score.py /app/

# Install any needed packages specified in requirements.txt
# Ensure you have a requirements.txt file with Flask and other dependencies
COPY requirements.txt /app/
COPY model_pipeline.pkl /app/
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]