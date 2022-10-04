# encoding:utf-8

import os

from fastapi import Request, File, FastAPI, Form
from fastapi.staticfiles import StaticFiles

from ai_modules import AiModules
from common import res_data, res_error, config, res_success, ndarray_to_base64str

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
    title='f-ai', version="2022.09.30", description="基于Paddle的模型接口",
    terms_of_service="https://github.com/2720851545/f-ai",
    contact={"name": "llyke", "url": "https://github.com/2720851545",
             "email": "2720851545@qq.com", },
    license_info={"name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
    openapi_tags=tags_metadata, )

ai_modules = AiModules(lazy_load=config.is_lazy_load_modules)


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
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.ch_pp_ocrv3(file)


@app.post("/animegan_v2_shinkai_53", tags=["image"], summary="图片转新海诚动漫风格")
async def req_animegan_v2_shinkai_53(request: Request, file: bytes = File(...)):
    """
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.animegan_v2_shinkai_53(file)


@app.post("/animegan_v2_hayao_99", tags=["image"], summary="图片转宫崎骏动漫风格")
async def req_animegan_v2_hayao_99(request: Request, file: bytes = File(...)):
    """
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.animegan_v2_hayao_99(file)


@app.post("/UGATIT_100w", tags=["image"], summary="人像动漫化")
async def req_UGATIT_100w(request: Request, file: bytes = File(...)):
    """
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.UGATIT_100w(file)


@app.post("/Photo2Cartoon", tags=["image"], summary="人像卡通化")
async def req_Photo2Cartoon(request: Request, file: bytes = File(...)):
    """
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.Photo2Cartoon(file)


@app.post("/U2Net", tags=["image"], summary="前景背景分割")
async def req_U2Net(request: Request, file: bytes = File(...)):
    """
        - mask: 背景图base64。
        - front: 前景图base64。
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.U2Net(file)


@app.post("/FCN_HRNet_W18_Face_Seg", tags=["image"], summary="人像分割")
async def req_FCN_HRNet_W18_Face_Seg(request: Request, file: bytes = File(...)):
    """
        - mask: 背景图base64。
        - face: 人像图base64。
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.FCN_HRNet_W18_Face_Seg(file)


@app.post("/ID_Photo_GEN", tags=["image"], summary="证件照生成")
async def req_ID_Photo_GEN(request: Request, file: bytes = File(...)):
    """
        - write: 白底的证件照base64。
        - blue: 蓝底的证件照base64。
        - red: 红底的证件照base64。
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.ID_Photo_GEN(file)


@app.post("/stgan_bald", tags=["image"], summary="图像生成1年、3年、5年的秃头效果")
async def req_stgan_bald(request: Request, file: bytes = File(...)):
    """
        - data_0: 秃头一年的预测结果图base64。
        - data_1: 秃头三年的预测结果图base64。
        - data_2: 秃头五年的预测结果图base64。
    """
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.stgan_bald(file)


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
    if os.getenv('F_AI_ENV') == 'test':
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
    if os.getenv('F_AI_ENV') == 'test':
        return res_error(message='服务器顶不住, 请本地运行测试😁')

    return ai_modules.chinese_ocr_db_crnn_mobile(file)


@app.post("/chinese_text_detection_db_server", tags=["image"], summary="文字位置识别")
async def req_chinese_text_detection_db_server(request: Request, file: bytes = File(...)):
    """
    检测文本框结果，文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
    """
    return ai_modules.chinese_text_detection_db_server(file)


@app.post("/pyramidbox_lite_server_mask", tags=["image"], summary="口罩检测(资源多, 效果较好)")
async def req_pyramidbox_lite_server_mask(request: Request, file: bytes = File(...)):
    """
        - label (str): 识别标签，为 'NO MASK' 或者 'MASK'
        - confidence (float): 识别的置信度
        - left (int): 边界框的左上角x坐标
        - top (int): 边界框的左上角y坐标
        - right (int): 边界框的右下角x坐标
        - bottom (int): 边界框的右下角y坐标
    """
    return ai_modules.pyramidbox_lite_server_mask(file)


@app.post("/pyramidbox_lite_mobile_mask", tags=["image"], summary="口罩检测(资源少, 效果较差)")
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


@app.post("/ernie_gen_lover_words", tags=["text"], summary="情话生成")
async def req_ernie_gen_lover_words(texts: str = Form(), beam_width: int = Form(5)):
    """
    - beam_width: 生成文本条数, 默认为5条

    输入情话开头(换行分隔)，输出情话下文
    """
    return ai_modules.ernie_gen_lover_words(texts.splitlines(), beam_width)


@app.post("/ernie_gen_poetry", tags=["text"], summary="诗歌生成")
async def req_ernie_gen_poetry(texts: str = Form(), beam_width: int = Form(5)):
    """
    beam_width: 生成文本条数, 默认为5条

    输入诗歌开头(换行分隔)，输出诗歌下文
    """
    return ai_modules.ernie_gen_poetry(texts.splitlines(), beam_width)


@app.post("/ernie_gen_couplet", tags=["text"], summary="对联生成")
async def req_ernie_gen_couplet(texts: str = Form(), beam_width: int = Form(5)):
    """
    - beam_width: 生成文本条数, 默认为5条

    输入上联文本(换行分隔)，输出下联文本
    """
    return ai_modules.ernie_gen_couplet(texts.splitlines(), beam_width)


@app.post("/ernie_vilg", tags=["text"], summary="文图生成, 生成文本描述内容的图像")
async def req_ernie_vilg(texts: str = Form(), style: str = Form('油画'), topk: int = Form(1)):
    """
    - texts: 输入的语句，描述想要生成的图像的内容(换行分隔)
    - style: 生成图像的风格，当前支持'油画','水彩','粉笔画','卡通','儿童画','蜡笔画','探索无限', 默认为'油画'
    - topk: 生成多少张图，最多生成6张, 默认为1条
    """
    return ai_modules.ernie_vilg(texts.splitlines(), style, topk)

if __name__ == '__main__':
    app.mount(
        "/static", StaticFiles(directory=config.image_result_path), name="static")
