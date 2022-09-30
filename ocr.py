# encoding:utf-8

from cmath import log
from warnings import catch_warnings

import cv2
import numpy as np
import paddlehub as hub
from common import *

ch_pp_ocrv3 = hub.Module(name="ch_pp-ocrv3")
hinese_text_detection_db_server = hub.Module(
    name="chinese_text_detection_db_server")
mask_detector = hub.Module(name="pyramidbox_lite_mobile_mask")
# ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效

image_result_path = '/tmp/ocr'


def ch_pp_ocrv3(file: bytes):
    try:
        result = ch_pp_ocrv3.recognize_text(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)], visualization=True,
            output_dir=image_result_path + '/ch_pp_ocrv3')
        result = result[0]['data']
        return res_success(data=result)
    except Exception as err:
        print('error', err)
        return res_error(message=str(err))


def chinese_text_detection_db_server(file: bytes):
    try:
        result = hinese_text_detection_db_server.detect_text(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)], visualization=True,
            output_dir=image_result_path + '/chinese_text_detection_db_server')
        result = result[0]['data']
        return res_success(data=result)
    except Exception as err:
        return res_error(message=str(err))


def pyramidbox_lite_mobile_mask(file: bytes):
    try:
        result = mask_detector.face_detection(
            images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)], visualization=True,
            output_dir=image_result_path + '/pyramidbox_lite_mobile_mask')
        result = result[0]['data']
        return res_success(data=result)
    except Exception as err:
        return res_error(message=str(err))
