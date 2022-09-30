
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
