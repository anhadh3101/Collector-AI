# Get the docker image and create the working directory for the app
FROM python:3.14
WORKDIR /usr/local/app

# Install dependencies
COPY requirements.txt ./
RUN pip install fastapi[standard]

# Copy the source code
COPY main.py ./

# Expose the port
EXPOSE 8000

# Set up an app user so the user doesn't run as root
RUN useradd app
USER app

# Run the server
CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0"]
