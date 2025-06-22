# Use official Python 3.10 image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (write inline for now)
COPY app.py ./
COPY models ./models
COPY image ./image
COPY measurement.csv ./

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask flask-cors mediapipe opencv-python tensorflow==2.19.0 keras==3.10.0 h5py python-multipart gunicorn

# Expose port
EXPOSE 8000

# Run the app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"] 