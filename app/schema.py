#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionOutput(BaseModel):
    """Image classification prediction (top-1 + optional top-k lists)."""
    label: str = Field(..., example="soybeans", title="Top-1 class label")
    confidence: float = Field(..., example=0.87, ge=0.0, le=1.0, title="Top-1 probability/confidence")
    top_labels: Optional[List[str]] = Field(
        None,
        example=["soybeans", "orange", "ginger"],
        title="Ordered list of top-k labels (descending by confidence)"
    )
    top_confidences: Optional[List[float]] = Field(
        None,
        example=[0.87, 0.08, 0.05],
        title="Probabilities/confidences corresponding to top_labels"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "label": "soybeans",
                "confidence": 0.87,
                "top_labels": ["soybeans", "orange", "ginger"],
                "top_confidences": [0.87, 0.08, 0.05]
            }
        }


class PredictionResponse(BaseModel):
    """Wrapper response returned by /predict."""
    error: bool = Field(False, example=False, title="Whether there is error")
    results: PredictionOutput = Field(..., title="Prediction payload")


class PlantCareResponse(BaseModel):
    """Response from the /plant-care endpoint with classification + care card."""
    error: bool = Field(False, example=False, title="Whether there is error")
    classification: PredictionOutput = Field(..., title="Image classification result")
    plant_care_card: dict = Field(..., title="Generated PlantCareCard from RAG pipeline")


class ErrorResponse(BaseModel):
    """Error response for the API."""
    error: bool = Field(..., example=True, title="Whether there is error")
    message: str = Field(..., example="", title="Error message")
    traceback: Optional[str] = Field(None, example="", title="Detailed traceback of the error")
