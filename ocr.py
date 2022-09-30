# encoding:utf-8

from cmath import log
import json
import paddlehub as hub
import cv2
import numpy as np

ocr = hub.Module(name="ch_pp-ocrv3")
# ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效


def img_ocr(file: bytes):
    # img_path = '/Users/liu/Downloads/healthcode/WechatIMG25.jpeg'
    result = ocr.recognize_text(
        images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)])
        
    # js = json.dumps(result, ensure_ascii=False)
    return result
