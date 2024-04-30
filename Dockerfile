FROM python:3.12.0

RUN apt-get update && apt-get install -y iputils-ping

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8502

CMD ["streamlit", "run", "app.py", "--server.port=8502"]
