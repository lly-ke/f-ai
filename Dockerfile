FROM paddlepaddle/paddle:2.3.2
EXPOSE 8000
# FROM continuumio/anaconda3 
# RUN pip install  -i https://mirror.baidu.com/pypi/simple paddlehub 

WORKDIR /f_ocr
COPY . /f_ocr

RUN pip install -i https://mirror.baidu.com/pypi/simple -r requirements.txt
ENTRYPOINT [ "uvicorn", "main:app", "--host 0.0.0.0", "--reload" ]
