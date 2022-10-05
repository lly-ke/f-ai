from fastapi import APIRouter, Request, File, FastAPI, Form
from f_ai.ai_modules import AiModules
from f_ai.common import  config, res_error

index_router = r = APIRouter()
ai_modules = AiModules(lazy_load=config.is_lazy_load_modules)

@r.post("/ch_pp-ocrv3", tags=["image"], summary="ocr文字识别(推荐)")
def req_ch_pp_ocrv3(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为：
    - text(str): 识别得到的文本
    - confidence(float): 识别文本结果置信度
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为空列表
    """

    return ai_modules.ch_pp_ocrv3(file)


@r.post("/animegan_v2_shinkai_53", tags=["image"], summary="图片转新海诚动漫风格")
def req_animegan_v2_shinkai_53(request: Request, file: bytes = File(...)):
    """
    """

    return ai_modules.animegan_v2_shinkai_53(file)


@r.post("/animegan_v2_hayao_99", tags=["image"], summary="图片转宫崎骏动漫风格")
def req_animegan_v2_hayao_99(request: Request, file: bytes = File(...)):
    """
    """

    return ai_modules.animegan_v2_hayao_99(file)


@r.post("/UGATIT_100w", tags=["image"], summary="人像动漫化")
def req_UGATIT_100w(request: Request, file: bytes = File(...)):
    """
    """

    return ai_modules.UGATIT_100w(file)


@r.post("/Photo2Cartoon", tags=["image"], summary="人像卡通化")
def req_Photo2Cartoon(request: Request, file: bytes = File(...)):
    """
    """

    return ai_modules.Photo2Cartoon(file)


@r.post("/U2Net", tags=["image"], summary="前景背景分割")
def req_U2Net(request: Request, file: bytes = File(...)):
    """
        - mask: 背景图base64。
        - front: 前景图base64。
    """

    return ai_modules.U2Net(file)


@r.post("/FCN_HRNet_W18_Face_Seg", tags=["image"], summary="人像分割")
def req_FCN_HRNet_W18_Face_Seg(request: Request, file: bytes = File(...)):
    """
        - mask: 背景图base64。
        - face: 人像图base64。
    """

    return ai_modules.FCN_HRNet_W18_Face_Seg(file)


@r.post("/ID_Photo_GEN", tags=["image"], summary="证件照生成")
def req_ID_Photo_GEN(request: Request, file: bytes = File(...)):
    """
        - write: 白底的证件照base64。
        - blue: 蓝底的证件照base64。
        - red: 红底的证件照base64。
    """

    return ai_modules.ID_Photo_GEN(file)


@r.post("/stgan_bald", tags=["image"], summary="图像生成1年、3年、5年的秃头效果")
def req_stgan_bald(request: Request, file: bytes = File(...)):
    """
        - data_0: 秃头一年的预测结果图base64。
        - data_1: 秃头三年的预测结果图base64。
        - data_2: 秃头五年的预测结果图base64。
    """

    return ai_modules.stgan_bald(file)


@r.post("/face_landmark_localization", tags=["image"], summary="人脸关键点检测")
def req_face_landmark_localization(request: Request, file: bytes = File(...)):
    """
    识别输入图片中的所有人脸关键点，每张人脸检测出68个关键点（人脸轮廓17个点，左右眉毛各5个点，左右眼睛各6个点，鼻子9个点，嘴巴20个点）
    - data: 图片中每张人脸的关键点坐标
    """

    return ai_modules.face_landmark_localization(file)


@r.post("/ultra_light_fast_generic_face_detector_1mb_640", tags=["image"], summary="人脸检测")
def req_ultra_light_fast_generic_face_detector_1mb_640(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为：
    - confidence (float): 识别的置信度
    - left (int): 边界框的左上角x坐标
    - top (int): 边界框的左上角y坐标
    - right (int): 边界框的右下角x坐标
    - bottom (int): 边界框的右下角y坐标
    """

    return ai_modules.ultra_light_fast_generic_face_detector_1mb_640(file)


@r.post("/chinese_ocr_db_crnn_server", tags=["image"], summary="CRNN汉字识别")
def req_ch_chinese_ocr_db_crnn_server(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为： 
    - text(str): 识别得到的文本 
    - confidence(float): 识别文本结果置信度 
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为[]
    """

    return ai_modules.chinese_ocr_db_crnn_server(file)


@r.post("/chinese_ocr_db_crnn_mobile", tags=["image"], summary="轻量级中文OCR")
def req_ch_chinese_ocr_db_crnn_mobile(request: Request, file: bytes = File(...)):
    """
    识别文本结果，列表中每一个元素为 dict，各字段为： 
    - text(str): 识别得到的文本 
    - confidence(float): 识别文本结果置信度 
    - text_box_position(list): 文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标 如果无识别结果则data为[]
    """

    return ai_modules.chinese_ocr_db_crnn_mobile(file)


@r.post("/chinese_text_detection_db_server", tags=["image"], summary="文字位置识别")
def req_chinese_text_detection_db_server(request: Request, file: bytes = File(...)):
    """
    检测文本框结果，文本框在原图中的像素坐标，4*2的矩阵，依次表示文本框左下、右下、右上、左上顶点的坐标
    """
    return ai_modules.chinese_text_detection_db_server(file)


@r.post("/pyramidbox_lite_server_mask", tags=["image"], summary="口罩检测(资源多, 效果较好)")
def req_pyramidbox_lite_server_mask(request: Request, file: bytes = File(...)):
    """
        - label (str): 识别标签，为 'NO MASK' 或者 'MASK'
        - confidence (float): 识别的置信度
        - left (int): 边界框的左上角x坐标
        - top (int): 边界框的左上角y坐标
        - right (int): 边界框的右下角x坐标
        - bottom (int): 边界框的右下角y坐标
    """
    return ai_modules.pyramidbox_lite_server_mask(file)


@r.post("/pyramidbox_lite_mobile_mask", tags=["image"], summary="口罩检测(资源少, 效果较差)")
def req_pyramidbox_lite_mobile_mask(request: Request, file: bytes = File(...)):
    """
        - label (str): 识别标签，为 'NO MASK' 或者 'MASK'
        - confidence (float): 识别的置信度
        - left (int): 边界框的左上角x坐标
        - top (int): 边界框的左上角y坐标
        - right (int): 边界框的右下角x坐标
        - bottom (int): 边界框的右下角y坐标
    """
    return ai_modules.pyramidbox_lite_mobile_mask(file)


@r.post("/senta_bilstm", tags=["text"], summary="情感分析")
def req_senta_bilstm(texts: str = Form()):
    """
    """
    return ai_modules.senta_bilstm(texts.splitlines())


@r.post("/porn_detection_lstm", tags=["text"], summary="文本涉黄预测")
def req_porn_detection_lstm(texts: str = Form()):
    """
    """
    return ai_modules.porn_detection_lstm(texts.splitlines())


@r.post("/ernie_gen_lover_words", tags=["text"], summary="情话生成")
def req_ernie_gen_lover_words(texts: str = Form(), beam_width: int = Form(5)):
    """
    - beam_width: 生成文本条数, 默认为5条

    输入情话开头(换行分隔)，输出情话下文
    """
    return ai_modules.ernie_gen_lover_words(texts.splitlines(), beam_width)


@r.post("/ernie_gen_poetry", tags=["text"], summary="诗歌生成")
def req_ernie_gen_poetry(texts: str = Form(), beam_width: int = Form(5)):
    """
    beam_width: 生成文本条数, 默认为5条

    输入诗歌开头(换行分隔)，输出诗歌下文
    """
    return ai_modules.ernie_gen_poetry(texts.splitlines(), beam_width)


@r.post("/ernie_gen_couplet", tags=["text"], summary="对联生成")
def req_ernie_gen_couplet(texts: str = Form(), beam_width: int = Form(5)):
    """
    - beam_width: 生成文本条数, 默认为5条

    输入上联文本(换行分隔)，输出下联文本
    """
    return ai_modules.ernie_gen_couplet(texts.splitlines(), beam_width)


@r.post("/ernie_vilg", tags=["text"], summary="文图生成, 生成文本描述内容的图像")
def req_ernie_vilg(texts: str = Form(), style: str = Form('油画'), topk: int = Form(1)):
    """
    - texts: 输入的语句，描述想要生成的图像的内容(换行分隔)
    - style: 生成图像的风格，当前支持'油画','水彩','粉笔画','卡通','儿童画','蜡笔画','探索无限', 默认为'油画'
    - topk: 生成多少张图，最多生成6张, 默认为1条
    """
    return ai_modules.ernie_vilg(texts.splitlines(), style, topk)
