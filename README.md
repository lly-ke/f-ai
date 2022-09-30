# 基于PaddleOCR的OCR识别接口

## 使用
### 第一种方式

- 下载docker-compose.yml,保存为docker-compose.yml
  
  https://raw.githubusercontent.com/2720851545/f-ocr/main/docker-compose.yml

- 进入docker-compose.yml文件夹,执行
`docker-compose up -d`

### 第二种方式

```bash
# 拉取仓库代码
git clone  https://github.com/2720851545/f-ocr.git

# 进入项目文件夹
cd f-ocr

# 运行服务
docker-compose up -d

# 修改代码重新构建
docker-compose up -d --build
# 或者
docker-compose build
```
### ui地址
- swagger ui地址: http://127.0.0.1:8000/docs

- redoc 文档地址: http://127.0.0.1:8000/redoc


### 注意事项


- 必须先安装docker和docker-compose

- docker-compose最好使用最新的, 下载地址: https://github.com/docker/compose/releases

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
