# Use the official Python image as a base
FROM python:3.12-slim

# Install system dependencies (removing git-lfs if not needed)
RUN apt-get update && apt-get install -y curl

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the model from the GitHub release
RUN mkdir -p /app/fake_news_detector && \
    curl -L -o /app/fake_news_detector/fake_news_detection_model.keras https://github.com/username/repository/releases/download/v1.0.0/fake_news_detection_model.keras

# Expose the port that the app will use
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "main:main", "--host", "0.0.0.0", "--port", "8000"]