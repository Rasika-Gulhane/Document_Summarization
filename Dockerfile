FROM python:3.11.3
RUN apt update -y && apt install awscli -y
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN pip install gunicorn
EXPOSE $PORT
CMD gunicorn --workers=4 --timeout 300 --bind 0.0.0.0:$PORT app:app