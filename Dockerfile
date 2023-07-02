FROM python:3.9.13-slim-buster

COPY manage.py

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]