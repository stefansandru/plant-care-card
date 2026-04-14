#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pydantic schemas for the PlantCareCard RAG pipeline."""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class LightingCondition(str, Enum):
    FULL_SUN = "full_sun"
    PARTIAL_SHADE = "partial_shade"
    SHADE = "shade"
    INDIRECT_LIGHT = "indirect_light"


class WateringFrequency(str, Enum):
    DAILY = "daily"
    EVERY_2_3_DAYS = "every_2_3_days"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class Difficulty(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"


class GrowthRate(str, Enum):
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"


class HumidityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WateringAdjustmentRule(BaseModel):
    condition: str = Field(
        description="Condition for adjustment, e.g., 'if temperature > 28°C' or 'if air is very dry'"
    )
    interval_days: Optional[int] = Field(
        None, description="Adjusted watering interval in days"
    )
    volume_l: Optional[float] = Field(
        None, description="Adjusted watering volume in liters"
    )
    note: Optional[str] = Field(
        None, description="Additional note for this adjustment rule"
    )


class PlantCareCard(BaseModel):
    """Structured plant care information card."""

    # Basic identification
    common_name: str = Field(description="Common name of the plant")
    latin_name: str = Field(description="Scientific/botanical Latin name")
    family: Optional[str] = Field(None, description="Plant family (e.g., Liliaceae)")
    native_habitat: str = Field(description="Describe the native habitat of the plant")
    summary: str = Field(
        description=(
            "Brief summary in up to 3 sentences about the plant native habitat "
            "and the minimum plant care."
        )
    )

    # Growing conditions
    outdoors: str = Field(description="Minimum outdoor growing conditions")
    indoors: str = Field(description="Minimum indoor growing conditions")
    lighting_conditions: List[LightingCondition] = Field(
        description="Suitable lighting conditions"
    )

    # Care requirements
    watering_interval_days: int = Field(
        description="How often to water, in days (e.g., 7 for weekly)"
    )
    watering_volume_l: Optional[float] = Field(
        None,
        description="Typical watering volume per session, in liters (e.g., 1.0)",
    )
    watering_adjustments: Optional[List[WateringAdjustmentRule]] = Field(
        None,
        description="Conditional rules for adjusting watering",
    )
    humidity_level: HumidityLevel = Field(
        description="Preferred humidity: low, medium, or high"
    )
    temperature_range_celsius: List[float] = Field(
        description="Optimal temperature range as [min, max] in Celsius, e.g., [18.0, 24.0]"
    )
    soil_type: str = Field(
        description="Preferred soil description"
    )

    # Maintenance
    difficulty_level: Difficulty = Field(description="Care difficulty level")
    fertilizer_needs: str = Field(
        description="Fertilization requirements (e.g., 'monthly during growing season')"
    )
    pruning_required: bool = Field(description="Whether regular pruning is needed")

    # Growth characteristics
    mature_height_cm: Optional[List[int]] = Field(
        None,
        description="Expected mature height range in centimeters as [min, max]",
    )
    growth_rate: GrowthRate = Field(
        description="Growth speed: slow, moderate, or fast"
    )

    # Seasonal information
    blooming_season: Optional[List[Season]] = Field(
        None, description="When the plant blooms"
    )
    planting_season: Optional[List[Season]] = Field(
        None, description="Best time to plant"
    )

    # Additional care
    toxicity: str = Field(description="Toxicity info for humans and pets")
    common_pests: List[str] = Field(
        default_factory=list,
        description="Common pests to watch for",
    )
    common_diseases: List[str] = Field(
        default_factory=list,
        description="Common diseases",
    )

    class Config:
        use_enum_values = True
