# encoding:utf-8

import os
import json

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from f_ai.common import config, res_error

from f_ai.routers.index import index_router

tags_metadata = [
    {
        "name": "image",
        "description": "图像处理",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://fastapi.tiangolo.com/",
        },
    },
    {
        "name": "text",
        "description": "文本处理",
    },
]
app = FastAPI(
    title='f-ai', version="2022.09.30", description="基于Paddle的模型接口",
    terms_of_service="https://github.com/2720851545/f-ai",
    contact={"name": "llyke", "url": "https://github.com/2720851545",
             "email": "2720851545@qq.com", },
    license_info={"name": "Apache 2.0",
                  "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
    openapi_tags=tags_metadata, )


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):

    if request.url.path.startswith("/api/") and os.getenv('F_AI_ENV') == 'test':
        return Response(
            content=json.dumps(
                res_error(message='服务器顶不住, 请本地运行测试😁'), indent=2),
            status_code=200,
        )

    response: Response = await call_next(request)

    return response


@app.get("/", summary="首页", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <h3>基于百度飞浆的人工智障服务</h3>
        <a href="https://f-ai-production.up.railway.r/docs">跳转官方文档</a>
    </html>
    """


app.include_router(
    index_router,
    prefix="/api/v1",
)

if __name__ == '__main__':
    app.mount(
        "/static", StaticFiles(directory=config.image_result_path), name="static")
