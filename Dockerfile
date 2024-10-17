# Use the official Python image as a base
FROM python:3.12-slim

# Install system dependencies (you can remove git-lfs if you're moving away from it)
RUN apt-get update && apt-get install -y git wget

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Download the model from the GitHub release
RUN mkdir -p /app/fake_news_detector && \
    wget -O /app/fake_news_detector/fake_news_detection_model.keras \
    https://github.com/anirudhrao20/Portfolio-ML-Models/releases/download/Model-Retrain-v1.0.2/fake_news_detection_model_1.0.2.keras

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]