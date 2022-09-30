# encoding:utf-8

from ocr import img_ocr

from fastapi import Request, File, FastAPI

app = FastAPI(
    title='字符识别', version="2022.09.30", description="基于PaddleOCR的OCR识别接口"    , terms_of_service="https://github.com/2720851545/f-ocr"    , contact={"name": "llyke", "url": "https://github.com/2720851545", "email": "2720851545@qq.com", }    
    , license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0.html"})


@app.get("/")
def read_root():
    return '基于百度飞浆的ocr服务'


@app.post("/ocr")
async def req_ocr_post(request: Request, file: bytes = File(...)):
    return img_ocr(file)
