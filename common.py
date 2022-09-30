# encoding:utf-8

# 识别结果保存
is_visualization = True
# 识别结果保存为位置
image_result_path = '/tmp/ocr'


def res_data(code=200, data=any, message="success") -> dict:
    return {
        'code': code,
        'message': message,
        'data': data,
    }


def res_error(message="error") -> dict:
    return res_data(code=500, message=message)


def res_success(data=any, message="success") -> dict:
    return res_data(code=200, message=message, data=data)
