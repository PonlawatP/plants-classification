from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np

from io import BytesIO
import base64

#load image as grayscale
def genHogData(img):
    img_data = base64.b64decode(img.split(',')[1])
    nparr = np.fromstring(img_data, np.uint8)

    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img_new = cv2.resize(img_gray, (128,128), cv2.INTER_AREA)
    win_Size = img_new.shape
    cell_size = (8,8)
    block_size = (16,16)
    block_stride = (8,8)
    num_bins = 9

    #set paramiters of hog descriptor
    hog = cv2.HOGDescriptor(win_Size,block_size,block_stride,cell_size,num_bins)
    
    #compute hog descriptor for gray scale image
    hog_descriptor = hog.compute(img_new)

    return hog_descriptor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get("/")
async def root():
    return {"message": "hello"}

@app.get("/api/genhog")
async def genhog(request: Request):
    data = await request.json()
    pre_data = genHogData(data['img'])
    return {"data": pre_data.tolist()}