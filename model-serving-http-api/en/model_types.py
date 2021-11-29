from pydantic import BaseModel, Field
from typing import Optional


class PersonName(BaseModel):
    first: str = Field(..., type=str, example="jane")
    last: Optional[str] = Field(type=Optional[str], example="shepard")
