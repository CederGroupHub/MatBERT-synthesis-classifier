import logging
from typing import List, Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse

from synthesis_classifier.model import get_model, get_tokenizer
from synthesis_classifier.pclassifier import run_batch

logger = logging.getLogger('Main')
logging.basicConfig(
    format="%(asctime)s [%(levelname)s][%(name)s]: %(message)s",
    datefmt="%m/%d %H:%M:%S",
    level=logging.INFO,
)


class ClassifierInput(BaseModel):
    paragraphs: List[str]


class ClassifierOutput(BaseModel):
    text: str
    tokens: List[str]
    scores: Dict[str, float]


class APIResponse(BaseModel):
    results: List[ClassifierOutput]


def get_response_json(text: List[str]):
    batch_size = 8
    results = []
    with torch.no_grad():
        for start in range(0, len(text), batch_size):
            end = min(start + batch_size, len(text))
            results.extend(run_batch(text[start:end], model, tokenizer))

    return JSONResponse({
        'results': results
    })


model = get_model()
tokenizer = get_tokenizer()
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/paragraph_classifier_batch", response_model=APIResponse)
def read_item(json_input: ClassifierInput):
    return get_response_json(json_input.paragraphs)


@app.get("/paragraph_classifier", response_model=ClassifierOutput)
def read_item(paragraph: str):
    result = run_batch([paragraph], model, tokenizer)[0]

    return JSONResponse(result)
