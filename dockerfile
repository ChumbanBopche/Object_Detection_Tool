# Dockerfile
# Use a Python base image that is slightly more robust for ML libraries
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
# This includes the full ultralytics/PyTorch stack
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and your model file
COPY . /app

# Expose the port (Flask/Gunicorn default is usually 8000 or 5000)
EXPOSE 8000

# Command to run your Flask app using Gunicorn
# Adjust app:app if your Flask app instance is named differently
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]