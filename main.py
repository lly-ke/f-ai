# encoding:utf-8

from ocr import img_ocr

from fastapi import Request, File, FastAPI

app = FastAPI(title='字符识别')


@app.get("/")
def read_root():
    return '基于百度飞浆的ocr服务'


@app.post("/ocr")
async def req_ocr_post(request: Request, file: bytes = File(...)):
    return img_ocr(file)
