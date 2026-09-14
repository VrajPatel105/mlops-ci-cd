FROM python:3.10

WORKDIR /app

COPY app/ /app/app/
COPY models/ /app/models/

RUN pip install --no-cache-dir -r /app/app/requirements.txt

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.main:app"]