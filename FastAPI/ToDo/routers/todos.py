# 文件：routers/todos.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session

from database import get_db
from table import DBToDo, User
# 【新增】：从 auth 部门请来我们的门禁保安
from routers.auth import get_current_user


class ToDoCreate(BaseModel):
    title: str
    completed: bool = False


class ToDoResponse(BaseModel):
    id: int
    title: str
    completed: bool
    owner_id: int | None = None

    class Config:
        from_attributes = True


router = APIRouter(prefix="/todos", tags=["待办事项"])


# 【修改 1】：强制查票！只有登录用户才能查列表，而且只能看到自己的！
@router.get("", response_model=List[ToDoResponse])
def get_all_todos(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)  # 👈 核心：只要加了这句，不带 Token 就进不来！
):
    # 用 filter 过滤，实现数据隔离！
    return db.query(DBToDo).filter(DBToDo.owner_id == current_user.id).all()


# 【修改 2】：创建事项时，自动打上创建者的思想钢印！
@router.post("", response_model=ToDoResponse)
def create_todo(
        todo: ToDoCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)  # 👈 强制查票
):
    db_todo = DBToDo(
        title=todo.title,
        completed=todo.completed,
        owner_id=current_user.id  # 👈 核心：这条记录永远属于当前登录的用户
    )
    db.add(db_todo)
    db.commit()
    db.refresh(db_todo)
    return db_todo


# 【修改 3】：删除时，不仅要查票，还要核对这个事项是不是你的！
@router.delete("/{todo_id}")
def delete_todo(
        todo_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)  # 👈 强制查票
):
    todo = db.query(DBToDo).filter(DBToDo.id == todo_id).first()
    if not todo:
        raise HTTPException(status_code=404, detail="找不到该事项")

    # 如果这个事项的 owner_id 和 当前登录人的 id 不一样，说明想删别人的数据！
    if todo.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="警告：你没有权限删除别人的数据！")

    db.delete(todo)
    db.commit()
    return {"message": "删除成功"}