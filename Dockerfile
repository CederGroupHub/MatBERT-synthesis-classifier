FROM python:3.6.12

WORKDIR /classification

# Install transformers
RUN pip3 install transformers
# Install fastAPI
RUN pip3 install -U fastapi[all]
# Install torch
RUN pip3 install torch
# Install mongo client
RUN pip3 install pymongo
# Install numpy/scipy
RUN pip3 install scipy
RUN pip3 install numpy

COPY . /classification
RUN pip3 install -e .
RUN python -m synthesis_classifier.model download

#uvicorn classification_api:app --port 8051 --reload --host 0.0.0.0