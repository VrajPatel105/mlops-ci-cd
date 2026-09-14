FROM python:3.10

WORKDIR /app

COPY app/ /docker_app/
COPY models/vectorizer.pkl /docker_app/models/vectorizer.pkl

RUN pip install --no-cache-dir -r /docker_app/requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["python", "/docker_app/main.py"]