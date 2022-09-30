# encoding:utf-8

from cmath import log
from warnings import catch_warnings

import cv2
import numpy as np
import paddlehub as hub
from common import *
from common import image_result_path, is_visualization

ch_pp_ocrv3_model = hub.Module(name="ch_pp-ocrv3")
chinese_text_detection_db_server_model = hub.Module(
    name="chinese_text_detection_db_server")
pyramidbox_lite_mobile_mask_model = hub.Module(name="pyramidbox_lite_mobile_mask")
senta_bilstm_model = hub.Module(name="senta_bilstm")
chinese_ocr_db_crnn_server_model = hub.Module(name="chinese_ocr_db_crnn_server")
chinese_ocr_db_crnn_mobile_model = hub.Module(name="chinese_ocr_db_crnn_mobile")

# ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效


def ch_pp_ocrv3(file: bytes):
    try:
        result = ch_pp_ocrv3_model.recognize_text(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
            visualization=is_visualization,
            output_dir=image_result_path + '/ch_pp_ocrv3')
        return res_success(data=result)
    except Exception as err:
        print('error', err)
        return res_error(message=str(err))

def chinese_ocr_db_crnn_server(file: bytes):
    try:
        result = chinese_ocr_db_crnn_server_model.recognize_text(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
            visualization=is_visualization,
            output_dir=image_result_path + '/chinese_ocr_db_crnn_server')
        return res_success(data=result)
    except Exception as err:
        print('error', err)
        return res_error(message=str(err))

def chinese_ocr_db_crnn_mobile(file: bytes):
    try:
        result = chinese_ocr_db_crnn_mobile_model.recognize_text(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
            visualization=is_visualization,
            output_dir=image_result_path + '/chinese_ocr_db_crnn_server')
        return res_success(data=result)
    except Exception as err:
        print('error', err)
        return res_error(message=str(err))


def chinese_text_detection_db_server(file: bytes):
    try:
        result = chinese_text_detection_db_server_model.detect_text(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
            visualization=is_visualization,
            output_dir=image_result_path + '/chinese_text_detection_db_server')
        return res_success(data=result)
    except Exception as err:
        return res_error(message=str(err))


def pyramidbox_lite_mobile_mask(file: bytes):
    try:
        result = pyramidbox_lite_mobile_mask_model.face_detection(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
            visualization=is_visualization,
            output_dir=image_result_path + '/pyramidbox_lite_mobile_mask')
        return res_success(data=result)
    except Exception as err:
        return res_error(message=str(err))

def senta_bilstm(texts: str):
    try:
        print(senta_bilstm)
        result = senta_bilstm_model.sentiment_classify(texts=texts,
                                           use_gpu=False,)
        return res_success(data=result)
    except Exception as err:
        return res_error(message=str(err))
