# encoding:utf-8

from distutils.command.config import config
import os
import base64, cv2
import distutils.util
from fastapi.logger import logger
import numpy as np

class Config: 
    def __init__(self) -> None:
        # 是否延迟加载AI模型
        self.is_lazy_load_modules = distutils.util.strtobool(os.getenv('IS_LAZY_LOAD_MODULES') if os.getenv('IS_LAZY_LOAD_MODULES') else 'false')
        logger.warning('是否延迟加载AI模型: %s', ('延迟加载' if self.is_lazy_load_modules else '启动加载'))

        # 是否保存识别结果图片
        self.is_visualization = distutils.util.strtobool(os.getenv('IS_VISUALIZATION') if os.getenv('IS_VISUALIZATION') else 'false')
        self.image_result_path = os.getcwd() + '/visualization'
        logger.warning('是否保存识别结果图片: %s', ('保存' if self.is_visualization else '不保存'))
        if self.is_visualization:
            # 图片识别结果保存位置
            self.image_result_path = os.getenv('IMAGE_RESULT_PATH')

            if not self.image_result_path:
                self.image_result_path = os.getcwd() + '/visualization'
                
            logger.warning('识别结果图片保存为位置: %s', self.image_result_path)
            if not os.path.exists(self.image_result_path):
                os.makedirs(self.image_result_path)


def ndarray_to_base64str(arr) -> dict:
    retval,img_buffer  = cv2.imencode('.png', arr)
    return 'data:image/png;base64,' + base64.b64encode(img_buffer).decode('utf-8')

def pil_image_to_base64str(arr) -> dict:
    return 'data:image/png;base64,' + base64.b64encode(base64.b64encode(np.asarray(arr))).decode('utf-8')

    
def res_data(code=200, data=None, message="success") -> dict:
    return {
        'code': code,
        'message': message,
        'data': data,
    }


def res_error(message="error") -> dict:
    return res_data(code=500, message=message)


def res_success(data=any, message="success") -> dict:
    return res_data(code=200, message=message, data=data)

config = Config()
