# For more information, please refer to https://aka.ms/vscode-docker-python
#FROM tensorflow/tensorflow:latest-py3
FROM python:3.8


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
ADD .. /app
ADD ../data /app/data
ADD ../data/target_dev /app/data/target_dev
ADD ../data/target_train /app/data/target_train
ADD ../data/non_target_dev /app/data/non_target_dev
ADD ../data/non_target_train /app/data/non_target_train

# During debugging, this entry point will be overridden. For more information, refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "src/detector.py"]
