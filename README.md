# AI模型http接口

- 官方示例: https://f-ai-production.up.railway.app/docs

- Apifox文档: https://www.apifox.cn/apidoc/shared-93cacff0-0ba9-467a-8c57-57680eeab8cb

## 已支持的模型

### ✅  图像处理模型POST
- ch_pp ocrv3 ocr文字识别(推荐)
- animegan_v2_shinkai_53 图片转新海诚动漫风格
- animegan_v2_hayao_99 图片转宫崎骏动漫风格
- UGATIT_100w 人像动漫化
- Photo2Cartoon 人像卡通化
- U2Net 前景背景分割
- FCN_HRNet_W18_Face_Seg 人像分割
- ID_Photo_GEN 证件照生成
- stgan_bald 图像生成1年、3年、5年的秃头效果
- face_landmark_localization 人脸关键点检测
- ultra_light_fast_generic_face_detector_1mb_640 人脸检测
- chinese_ocr_db_crnn_server CRNN汉字识别
- chinese_ocr_db_crnn_mobile 轻量级中文OCR
- chinese_text_detection_db_server 文字位置识别
- pyramidbox_lite_server_mask 口罩检测(资源多, 效果较好)
- pyramidbox_lite_mobile_mask 口罩检测(资源少, 效果较差)

## ✅  文本处理模型 
- senta_bilstm 情感分析
- porn_detection_lstm 文本涉黄预测
- ernie_gen_lover_words 情话生成
- ernie_gen_poetry 诗歌生成
- ernie_gen_couplet 对联生成
- ernie_vilg 文图生成, 生成文本描述内容的图像

## 使用
### 第一种方式

- 下载docker-compose.yml,保存为docker-compose.yml
  
  https://raw.githubusercontent.com/2720851545/f-ai/main/docker-compose.yml

- 进入docker-compose.yml文件夹,执行
`docker-compose up -d`

### 第二种方式

```bash
# 拉取仓库代码
git clone  https://github.com/2720851545/f-ai.git

# 进入项目文件夹
cd f-ai

# 运行服务
docker-compose up -d

# 修改代码重新构建
docker-compose up -d --build
# 或者
docker-compose build
```
### 暴露地址
- swagger ui地址: http://127.0.0.1:8000/docs
- redoc 文档地址: http://127.0.0.1:8000/redoc
- 识别结果图片地址: http://127.0.0.1:8000/static/xx/xxx.jpg 

### 注意事项


- 必须先安装docker和docker-compose

- docker-compose最好使用最新的, 下载地址: https://github.com/docker/compose/releases

- 多个模型不支持同时并发访问, 请分开使用

## 开发说明

```bash
# 创建隔离环境
conda create --name paddle_env python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  

# 安装百度飞浆
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlehub -i https://mirror.baidu.com/pypi/simple

# 安装依赖
pip install -f requirements.txt
```
### 开发注意事项

pip安包可能会提示已存在，可以使用

```bash
pip install 包名 --target=第三方包目录
# 例如
pip install fastapi --target=/usr/local/anaconda3/envs/paddle_env/lib/python3.8/site-packages
```


## 社区
- Discord: https://discord.gg/yzfKySW2zx
