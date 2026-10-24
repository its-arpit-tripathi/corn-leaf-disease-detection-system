from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: EmailStr
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class DiseaseInfo(BaseModel):
    title: str
    disease_name: str = ""
    crop_name: str = "Maize (Corn)"
    pathogen: str = ""
    severity: str = ""
    status: str = "Diseased"
    summary: str = ""
    symptoms: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)


class PredictionOut(BaseModel):
    id: int
    filename: str
    predicted_class: str
    disease_name: str = ""
    crop_name: str = "Maize (Corn)"
    status: str = "Diseased"  # Diseased | Healthy | Not Maize
    pathogen: str = ""
    severity: str = ""
    label: str
    confidence: float
    probabilities: dict[str, float]
    is_corn_leaf: bool
    is_maize: bool = True
    low_confidence: bool = False
    rejection_reason: str | None = None
    suspected_disease: str | None = None
    info: DiseaseInfo
    created_at: datetime


class StatsOut(BaseModel):
    total: int
    healthy: int
    diseased: int
    not_leaf: int = 0
    by_class: dict[str, int]
    average_confidence: float
    last_7_days: list[dict]
