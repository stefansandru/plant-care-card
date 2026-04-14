#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import traceback
import contextlib
from joblib import load
import json

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from io import BytesIO
from PIL import Image
import numpy as np
import torch

from model import Model
from predict import predict
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *
from util import abs_path
from rag_pipeline import generate_plant_care_card


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize FastAPI on startup.
    """
    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model = Model()
    model.load_state_dict(torch.load(
        abs_path(CONFIG['MODEL_PATH']), map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    with open(abs_path(CONFIG['CLASS_MAP_PATH']), 'r') as f:
        class_map = json.load(f)

    # add model and other preprocess tools to app state
    app.package = {
        "class_map": class_map,
        "model": model
    }

    # Set RAG pipeline env vars from config (Mistral API + Tavily)
    if CONFIG.get('MISTRAL_API_KEY'):
        os.environ.setdefault('MISTRAL_API_KEY', CONFIG['MISTRAL_API_KEY'])
    if CONFIG.get('TAVILY_API_KEY'):
        os.environ.setdefault('TAVILY_API_KEY', CONFIG['TAVILY_API_KEY'])
    if CONFIG.get('LLM_MODEL'):
        os.environ.setdefault('LLM_MODEL', CONFIG['LLM_MODEL'])

    rag_ready = bool(os.environ.get('MISTRAL_API_KEY')) and bool(os.environ.get('TAVILY_API_KEY'))
    logger.info(f'RAG pipeline ready: {rag_ready}')

    yield  # server is running


# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Description of the ML Model",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None,
    lifespan=lifespan,
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.post('/api/v1/predict',
          response_model=PredictionResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def do_predict(file: UploadFile = File(...)): # changed to async
    """Image classification prediction endpoint.

    Accepts a single uploaded image file (multipart/form-data) and returns
    top-1 prediction plus optional ordered probability lists.
    """
    logger.info('API predict called')
    logger.info(f'input: {file.filename}, content_type: {file.content_type}')

    # Basic content-type guard (optional)
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'):
        return JSONResponse(status_code=422, content={
            "error": True,
            "message": f"Unsupported content-type: {file.content_type}",
            "traceback": None
        })

    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=422, content={
            "error": True,
            "message": f"Failed to decode image: {e}",
            "traceback": None
        })

    probs = predict(app.package, img)

    class_map = app.package.get("class_map", {})

    # Top-1: Direct lookup like in notebook
    top_idx = int(np.argmax(probs))
    top_label = class_map.get(str(top_idx), f"unknown_class_{top_idx}")
    top_conf = float(round(float(probs[top_idx]), CONFIG['ROUND_DIGIT']))

    # Top-k lists: all predictions sorted by confidence
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [class_map.get(str(i), f"unknown_{i}") for i in sorted_indices]
    sorted_confs = [float(round(float(probs[i]), CONFIG['ROUND_DIGIT'])) for i in sorted_indices]


    result_payload = {
        "label": top_label,
        "confidence": top_conf,
        "top_labels": sorted_labels,
        "top_confidences": sorted_confs
    }

    logger.info(f'results: {result_payload}')

    return {"error": False, "results": result_payload}


@app.post('/api/v1/plant-care',
          response_model=PlantCareResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
async def do_plant_care(file: UploadFile = File(...)):
    """Plant care card endpoint.

    Accepts a plant image, classifies it with EfficientNet-B1,
    then runs the Mistral RAG pipeline to generate a PlantCareCard.
    """
    logger.info('API plant-care called')
    logger.info(f'input: {file.filename}, content_type: {file.content_type}')

    # Validate API keys are set
    if not os.environ.get('MISTRAL_API_KEY') or not os.environ.get('TAVILY_API_KEY'):
        return JSONResponse(status_code=500, content={
            "error": True,
            "message": "RAG pipeline not configured. Set MISTRAL_API_KEY and TAVILY_API_KEY environment variables.",
            "traceback": None
        })

    # Content-type guard
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'):
        return JSONResponse(status_code=422, content={
            "error": True,
            "message": f"Unsupported content-type: {file.content_type}",
            "traceback": None
        })

    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=422, content={
            "error": True,
            "message": f"Failed to decode image: {e}",
            "traceback": None
        })

    # Step 1: Classify the plant
    probs = predict(app.package, img)
    class_map = app.package.get("class_map", {})
    top_idx = int(np.argmax(probs))
    top_label = class_map.get(str(top_idx), f"unknown_class_{top_idx}")
    top_conf = float(round(float(probs[top_idx]), CONFIG['ROUND_DIGIT']))

    classification = {
        "label": top_label,
        "confidence": top_conf,
    }
    logger.info(f'Classification: {top_label} ({top_conf})')

    # Step 2: Generate PlantCareCard via RAG pipeline
    try:
        max_revisions = CONFIG.get('MAX_REVISIONS', 2)
        care_card = generate_plant_care_card(top_label, max_revisions=max_revisions)
        card_dict = care_card.model_dump()
    except Exception as e:
        logger.error(f'RAG pipeline error: {e}')
        return JSONResponse(status_code=500, content={
            "error": True,
            "message": f"Failed to generate plant care card: {e}",
            "traceback": str(e)
        })

    return {
        "error": False,
        "classification": classification,
        "plant_care_card": card_dict,
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, 
                # log_config=abs_path("log.ini")
                )
