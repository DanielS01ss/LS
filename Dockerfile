FROM python:3.11-slim

WORKDIR /app

COPY . .

EXPOSE 8000

RUN pip install -r requirements.txt

CMD python3 main.py

