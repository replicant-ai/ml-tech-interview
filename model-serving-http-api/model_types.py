from pydantic import BaseModel, Field
from typing import Optional


class PersonCountryOfBirth(BaseModel):
    first_name: str = Field(..., type=str, example="Jane")
    last_name: Optional[str] = Field(type=Optional[str], example="Shepard")
    country_of_birth: str = Field(..., type=str, example="Canada")
    score: float = Field(default=1.0, type=float, example=1.0)
