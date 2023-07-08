from fastapi import FastAPI, Request, File, UploadFile
from http import HTTPStatus 
from typing import Dict
from functools import wraps

import numpy as np

from activity_classifier.config import *

from tensorflow.keras.models import load_model
import logging

from activity_classifier.load_experiment import get_artifacts
from activity_classifier import predict

from PIL import Image
from io import BytesIO

app = FastAPI(
        title="Cooking Activity Classification",
        description = "Classify the cooking activity performed from images",
        version=0.1

)


def construct_response(f):

    @wraps(f)
    async def wrap(request:Request, *args, **kwargs):
        results = await f(request, *args, **kwargs)
        response = {}

        response['message'] = results['message']

        response['status-code'] = results['status-code']
        response['method'] = request.method
        response['url'] = request.url._url

        if 'data' in results:
            response['data'] = results['data']
        return response


    return wrap


@app.get("/")
@construct_response
def _index(request: Request) -> Dict:

    response = {
                    "message": HTTPStatus.OK.phrase,
                    "status-code": HTTPStatus.OK,
                    "data": {}

                }
    return response



@app.on_event("startup")
def load_artifacts():

    global artifacts

    artifacts = get_artifacts("baselines_3")

    logging.info("Model ready for inference")


@app.get("/performance", tags=["performance"])
@construct_response
def _performance(request:Request, filter:str=None) -> Dict:
    performance = artifacts['performance']
    logging.info(performance)

    data = {"performance": performance.get(filter, performance)}

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data" : data
    }
    return response


@app.post("/predict", tags=["predictions"])
@construct_response
async def _predict(request: Request, image: UploadFile=File(...)):
    
    image_bytes = await image.read()
    image = Image.open(BytesIO(image_bytes))
    image = np.array(image)

    label = predict.predict(image, artifacts)

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": label
    }
    return response
