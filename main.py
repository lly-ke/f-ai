# encoding:utf-8

import os

from fastapi import Request, File, FastAPI, Form
from fastapi.staticfiles import StaticFiles

from ai_modules import AiModules
from common import image_result_path, res_error


tags_metadata = [
    {
        "name": "image",
        "description": "图像处理",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
    {
        "name": "text",
        "description": "文本处理",
    },
]
app = FastAPI(
    title='f-ocr', version="2022.09.30", description="基于Paddle的模型接口",
    terms_of_service="https://github.com/2720851545/f-ocr",
    contact={"name": "llyke", "url": "https://github.com/2720851545", "email": "2720851545@qq.com", },
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
    openapi_tags=tags_metadata, )

app.mount("/static", StaticFiles(directory=image_result_path), name="static")

ai_modules = AiModules()

@app.get("/", summary="首页")
def read_root():
    return '基于百度飞浆的人工智障服务'


@app.post("/ch_pp-ocrv3", tags=["image"], summary="ocr文字识别(推荐)")
async def req_ch_pp_ocrv3(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为：
    - text(str): 识别得到的文本
    - confidence(float): 识别文本结果置信度
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为空列表
    """
    if os.getenv('F_OCR_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.ch_pp_ocrv3(file)


@app.post("/face_landmark_localization", tags=["image"], summary="人脸关键点检测")
async def req_face_landmark_localization(request: Request, file: bytes = File(...)):
    """
    识别输入图片中的所有人脸关键点，每张人脸检测出68个关键点（人脸轮廓17个点，左右眉毛各5个点，左右眼睛各6个点，鼻子9个点，嘴巴20个点）
    - data: 图片中每张人脸的关键点坐标
    """

    return ai_modules.face_landmark_localization(file)

@app.post("/ultra_light_fast_generic_face_detector_1mb_640", tags=["image"], summary="人脸检测")
async def req_ultra_light_fast_generic_face_detector_1mb_640(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为：
    - confidence (float): 识别的置信度
    - left (int): 边界框的左上角x坐标
    - top (int): 边界框的左上角y坐标
    - right (int): 边界框的右下角x坐标
    - bottom (int): 边界框的右下角y坐标
    """

    return ai_modules.ultra_light_fast_generic_face_detector_1mb_640(file)

@app.post("/chinese_ocr_db_crnn_server", tags=["image"], summary="CRNN汉字识别")
async def req_ch_chinese_ocr_db_crnn_server(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为： 
    - text(str): 识别得到的文本 
    - confidence(float): 识别文本结果置信度 
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为[]
    """
    if os.getenv('F_OCR_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.chinese_ocr_db_crnn_server(file)


@app.post("/chinese_ocr_db_crnn_mobile", tags=["image"], summary="轻量级中文OCR")
async def req_ch_chinese_ocr_db_crnn_mobile(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为： 
    - text(str): 识别得到的文本 
    - confidence(float): 识别文本结果置信度 
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为[]
    """
    if os.getenv('F_OCR_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.chinese_ocr_db_crnn_mobile(file)


@app.post("/chinese_text_detection_db_server", tags=["image"], summary="文字位置识别")
async def req_chinese_text_detection_db_server(request: Request, file: bytes = File(...)):
    """
    检测文本框结果，文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
    """
    return ai_modules.chinese_text_detection_db_server(file)


@app.post("/pyramidbox_lite_mobile_mask", tags=["image"], summary="口罩检测")
async def req_pyramidbox_lite_mobile_mask(request: Request, file: bytes = File(...)):
    """
        - label (str): 识别标签，为 'NO MASK' 或者 'MASK'
        - confidence (float): 识别的置信度
        - left (int): 边界框的左上角x坐标
        - top (int): 边界框的左上角y坐标
        - right (int): 边界框的右下角x坐标
        - bottom (int): 边界框的右下角y坐标
    """
    return ai_modules.pyramidbox_lite_mobile_mask(file)


@app.post("/senta_bilstm", tags=["text"], summary="情感分析")
async def req_senta_bilstm(texts: str = Form()):
    """
    """
    return ai_modules.senta_bilstm(texts.splitlines())


@app.post("/porn_detection_lstm", tags=["text"], summary="文本涉黄预测")
async def req_porn_detection_lstm(texts: str = Form()):
    """
    """
    return ai_modules.porn_detection_lstm(texts.splitlines())
