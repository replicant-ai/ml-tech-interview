"""
HTTP API that serves ML models predicting country of birth by name.
Supports multiple languages.

"""
from utils import infer
from en.model_types import PersonName
from fr.model_types import PersonNameFR
from en.model import load_model_en
from fr.model import load_model_fr
from model_types import PersonCountryOfBirth
from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.post("/getCountryOfBirthByNameEn", response_model=PersonCountryOfBirth)
async def getCountryOfBirthByNameEn(name: PersonName):
    model_en, _ = load_model_en()
    with infer(name.first, model_en) as p:
        return PersonCountryOfBirth(first_name=name.first, last_name=name.last, country_of_birth=p.country_of_birth)


@app.post("/getCountryOfBirthByNameFR", response_model=PersonCountryOfBirth)
async def getCountryOfBirthByNameFR(name: PersonNameFR):
    model_fr = load_model_fr()
    with infer(name.first, model_fr) as p:
        return PersonCountryOfBirth(first_name=name.first, last_name=name.last, country_of_birth=p.country_of_birth)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=False,
        loop="uvloop",
    )
