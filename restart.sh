#!/bin/bash

docker kill agenh

docker rm agenh

docker build -t agenh .

docker run -p 8000:8000 --name agenh -it --entrypoint bash -d agenh  

docker exec -it agenh python -m ipykernel install --user --name=myvirtualenv

docker exec -it agenh python manage.py runserver 0.0.0.0:8000
