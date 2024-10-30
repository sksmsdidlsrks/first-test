import numpy as np
import cv2
import easyocr
from fastapi import FastAPI, File, UploadFile

reader = easyocr.Reader(['ch_sim','en'])

app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contens = await file.read()
    binary = np.fromstring(contens, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    


     # this needs to run only once to load the model into memory
    result = reader.readtext(cv_mat)
    

    texts = [{"text": item[1], "confidence": item[2]} for item in result]


    return {"filename": texts}