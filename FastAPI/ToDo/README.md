# 📝FastAPI 实战：带 JWT 鉴权的待办事项系统

## 1. 项目简介
&emsp;&emsp;本项目是一个基于 FastAPI 框架开发的轻量级、企业级规范的前后端分离待办事项（To-Do List）系统。
项目实现了完整的用户注册、登录、密码哈希加密、JWT 身份验证，以及基于用户的多租户数据隔离机制。
## 2. 技术栈
![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![JWT](https://img.shields.io/badge/JWT-Security-black?logo=JSON%20web%20tokens)
![License](https://img.shields.io/badge/License-MIT-green)


## 3. 后端目录结构

```text
ToDo/
├── database.py       # 数据库基础连接配置 (Engine, Session)
├── table.py          # 数据库表结构模型 (User表, DBToDo表及外键关系)
├── main.py           # 核心应用入口 (挂载路由、启动建表、CORS配置)
├── routers/          # 路由模块目录 (部门经理)
│   ├── auth.py       # 认证路由：处理注册、登录、Token签发及鉴权拦截器
│   └── todos.py      # 业务路由：处理待办事项的CRUD操作(需验证Token)
└── index.html        # 前端可视化测试台