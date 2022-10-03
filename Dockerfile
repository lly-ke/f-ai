FROM paddlepaddle/paddle:2.3.2
EXPOSE 8000
# FROM continuumio/anaconda3 
# RUN pip install  -i https://mirror.baidu.com/pypi/simple paddlehub

WORKDIR /f-ai
# docker build缓存
COPY ./requirements.txt /f-ai
RUN pip install -i https://mirror.baidu.com/pypi/simple -r requirements.txt
RUN pip install -I shapely pyclipper
ENV TZ Asia/Shanghai
# RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# USER f-ai
COPY . /f-ai

ENTRYPOINT python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
