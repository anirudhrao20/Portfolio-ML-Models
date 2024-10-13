# Use the official Python image as a base
FROM python:3.12-slim

# Install system dependencies (you can remove git-lfs if you're moving away from it)
RUN apt-get update && apt-get install -y git-lfs

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Git LFS and pull any large files tracked by it
RUN git lfs install
RUN git lfs pull

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]