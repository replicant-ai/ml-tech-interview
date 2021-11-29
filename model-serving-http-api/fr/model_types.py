from pydantic import BaseModel, Field
from typing import Optional


class PersonNameFR(BaseModel):
    first: str = Field(..., type=str, example="jeanne")
    last: Optional[str] = Field(type=Optional[str], example="d'arc")
