# encoding:utf-8

import os
import distutils.util
from fastapi.logger import logger

# 是否延迟加载AI模型
is_lazy_load_modules = None
is_lazy_load_modules = distutils.util.strtobool(os.getenv('IS_LAZY_LOAD_MODULES') if os.getenv('IS_LAZY_LOAD_MODULES') else 'false')
logger.warning('是否延迟加载AI模型: %s', ('延迟加载' if is_lazy_load_modules else '启动加载'))

# 是否保存识别结果图片
image_result_path = None
is_visualization = distutils.util.strtobool(os.getenv('IS_VISUALIZATION') if os.getenv('IS_VISUALIZATION') else 'false')
logger.warning('是否保存识别结果图片: %s', ('保存' if is_visualization else '不保存'))
if is_visualization:
    # 图片识别结果保存位置
    image_result_path = os.getenv('IMAGE_RESULT_PATH')
    if image_result_path == None:
        image_result_path = '/tmp/ocr'
    logger.warning('识别结果图片保存为位置: %s', image_result_path)
    if not os.path.exists(image_result_path):
        os.makedirs(image_result_path)

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
