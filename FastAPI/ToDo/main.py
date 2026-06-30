from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import todos

app = FastAPI(title="代办事项 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(todos.router)

@app.get("/")
def root():
    return {"message": "运行正常"}