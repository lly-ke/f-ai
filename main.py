# encoding:utf-8

import ocr

from fastapi import Request, File, FastAPI

app = FastAPI(
    title='字符识别', version="2022.09.30", description="基于PaddleOCR的OCR识别接口",
    terms_of_service="https://github.com/2720851545/f-ocr",
    contact={"name": "llyke", "url": "https://github.com/2720851545", "email": "2720851545@qq.com", }
    , license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0.html"})


@app.get("/", summary="首页")
def read_root():
    return '基于百度飞浆的ocr服务'


@app.post("/ch_pp-ocrv3", summary="ocr文字识别")
async def reqch_pp_ocrv3(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为：
    - text(str): 识别得到的文本
    - confidence(float): 识别文本结果置信度
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为空列表
    """
    return ocr.ch_pp_ocrv3(file)


@app.post("/chinese_text_detection_db_server", summary="文字位置识别")
async def req_chinese_text_detection_db_server(request: Request, file: bytes = File(...)):
    """
    检测文本框结果，文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
    """
    return ocr.chinese_text_detection_db_server(file)


@app.post("/pyramidbox_lite_mobile_mask", summary="口罩检测")
async def req_pyramidbox_lite_mobile_mask(request: Request, file: bytes = File(...)):
    """
        - label (str): 识别标签，为 'NO MASK' 或者 'MASK'
        - confidence (float): 识别的置信度
        - left (int): 边界框的左上角x坐标
        - top (int): 边界框的左上角y坐标
        - right (int): 边界框的右下角x坐标
        - bottom (int): 边界框的右下角y坐标
    """
    return ocr.pyramidbox_lite_mobile_mask(file)
