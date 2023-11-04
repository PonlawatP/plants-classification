import requests
import pickle
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np

plant_type = {
    0: "kale",
    1: "sweetpotatoes",
    2: "spinach",
    3: "peperchili",
    4: "classname.txt",
    5: "tobacco",
    6: "guava",
    7: "bilimbi",
    8: "aloevera",
    9: "cucumber",
    10: "pomelo",
    11: "watermelon",
    12: "cantaloupe",
    13: "waterapple",
    14: "curcuma",
    15: "mango",
    16: "coconut",
    17: "galangal",
    18: "eggplant",
    19: "cassava",
    20: "corn",
    21: "soybeans",
    22: "banana",
    23: "longbeans",
    24: "ginger",
    25: "paddy",
    26: "papaya",
    27: "shallot",
    28: "pineapple",
    29: "melon",
    30: "orange"
}

# api_hoggen = "http://localhost:8000/api/genhog"
api_hoggen = "http://ai-hoggen:8000/api/genhog"

model = pickle.load(open(f'model/plantsModel.pk', 'rb'))

def predict_plantType(mdl, HOG):
    pred = mdl.predict([HOG])
    return plant_type[pred[0]]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)


@app.post("/api/plantguess")
async def genhog(request: Request):
    data = await request.json()

    try:
        hog_resp = requests.get(api_hoggen, json={"img": data['img']},headers={"Content-Type": "application/json"})
        sec_data = hog_resp.json()
        res = predict_plantType(model, sec_data['data'])
        return {"result": res}
    except:
        raise HTTPException(status_code=500, detail="invalid value")
    

app.mount('/', StaticFiles(directory='app/web', html=True),name='static')