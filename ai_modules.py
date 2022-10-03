# encoding:utf-8

from fastapi.logger import logger
import cv2
import numpy as np
import paddlehub as hub

from common import *
from common import image_result_path, is_visualization

class AiModules:
    def __init__(self, lazy_load=False):
        if lazy_load == False:
            self.ch_pp_ocrv3_model = hub.Module(name="ch_pp-ocrv3")
            self.chinese_text_detection_db_server_model = hub.Module(
                name="chinese_text_detection_db_server")
            self.pyramidbox_lite_mobile_mask_model = hub.Module(name="pyramidbox_lite_mobile_mask")
            self.senta_bilstm_model = hub.Module(name="senta_bilstm")
            self.chinese_ocr_db_crnn_server_model = hub.Module(name="chinese_ocr_db_crnn_server")
            self.chinese_ocr_db_crnn_mobile_model = hub.Module(name="chinese_ocr_db_crnn_mobile")
            self.porn_detection_lstm_model = hub.Module(name="porn_detection_lstm")
            self.ultra_light_fast_generic_face_detector_1mb_640_model = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
            self.face_landmark_localization_model = hub.Module(name="face_landmark_localization")

            # ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)       # mkldnn加速仅在CPU下有效


    def face_landmark_localization(self, file: bytes):
        try:
            result = self.face_landmark_localization_model.keypoint_detection(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/face_landmark_localization')
            return res_success(data=result)
        except Exception as err:
            print('error', err)
            return res_error(message=str(err))
    def ultra_light_fast_generic_face_detector_1mb_640(self, file: bytes):
        try:
            result = self.ultra_light_fast_generic_face_detector_1mb_640_model.face_detection(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/ultra_light_fast_generic_face_detector_1mb_640')
            return res_success(data=result)
        except Exception as err:
            print('error', err)
            return res_error(message=str(err))

    def ch_pp_ocrv3(self, file: bytes):
        try:
            result = self.ch_pp_ocrv3_model.recognize_text(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/ch_pp_ocrv3')
            return res_success(data=result)
        except Exception as err:
            print('error', err)
            return res_error(message=str(err))

    def chinese_ocr_db_crnn_server(self, file: bytes):
        try:
            result = self.chinese_ocr_db_crnn_server_model.recognize_text(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/chinese_ocr_db_crnn_server')
            return res_success(data=result)
        except Exception as err:
            print('error', err)
            return res_error(message=str(err))


    def porn_detection_lstm(self, texts: str):
        try:
            result = self.porn_detection_lstm_model.detection(texts=texts)
            return res_success(data=result)
        except Exception as err:
            print('error', err)
            return res_error(message=str(err))

    def chinese_ocr_db_crnn_mobile(self, file: bytes):
        try:
            result = self.chinese_ocr_db_crnn_mobile_model.recognize_text(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/chinese_ocr_db_crnn_server')
            return res_success(data=result)
        except Exception as err:
            print('error', err)
            return res_error(message=str(err))


    def chinese_text_detection_db_server(self, file: bytes):
        try:
            result = self.chinese_text_detection_db_server_model.detect_text(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/chinese_text_detection_db_server')
            return res_success(data=result)
        except Exception as err:
            return res_error(message=str(err))


    def pyramidbox_lite_mobile_mask(self, file: bytes):
        try:
            result = self.pyramidbox_lite_mobile_mask_model.face_detection(
                images=[cv2.imdecode(np.frombuffer(file, dtype=np.uint8), cv2.IMREAD_COLOR)],
                visualization=is_visualization,
                output_dir=image_result_path + '/pyramidbox_lite_mobile_mask')
            return res_success(data=result)
        except Exception as err:
            return res_error(message=str(err))

    def senta_bilstm(self, texts: str):
        try:
            result = self.senta_bilstm_model.sentiment_classify(texts=texts,
                                            use_gpu=False,)
            return res_success(data=result)
        except Exception as err:
            return res_error(message=str(err))

    def __getattr__(self, name):
        if ((name in dir(self)) == False) and name.endswith('_model'):
            model_name = name[::-1].replace('_model'[::-1], '', 1)[::-1]
            logger.warn('加载%s模型', model_name)
            self.__setattr__(name, hub.Module(name=model_name))
        return self.__getattribute__(name)

if __name__ == '__main__':
    ai = AiModules(True)
    print(ai.senta_bilstm(['今天一直在下雨']))
    print(ai.senta_bilstm(['今天一直在下雨']))
