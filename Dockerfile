FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app
COPY ./examples/predict_fastapi.py /app/main.py
RUN pip install -e /app
RUN python -m synthesis_classifier.model download
