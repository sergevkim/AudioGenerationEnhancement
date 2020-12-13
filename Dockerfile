# pull official base image TODO add cuda support
FROM python 

RUN pip install pipenv
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --system --deploy

# set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# set work directory
WORKDIR /project

# copy everything else
ADD api api
ADD setup.py setup.py
ADD agenh agenh
ADD manage.py manage.py

RUN pip install -e . && rm -rf agenh.egg-info

#RUN python manage.py runserver 0.0.0.0:8000
