import uvicorn
from  fastapi import FastAPI
from pydantic import BaseModel

app=FastAPI()

class input(BaseModel):
        brand: str
        model: str
        color: str
        year: int
        power_kw: float
        power_ps: float
        transmission_type: str
        fuel_type: str
        fuel_consumption_l_100km: float
        fuel_consumption_g_km: float
        mileage_in_km: float




@app.post('/prediction')




























