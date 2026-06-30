from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy.orm import Session


from database import get_db
from table import DBToDo

class ToDoCreate(BaseModel):
    title: str
    completed: bool = False

class ToDoResponse(BaseModel):
    id: int
    title: str
    completed: bool
    owner_id: int | None = None  # 顺便加个字段，以后前端能看到是谁创建的

    class Config:
        from_attributes = True

router = APIRouter(prefix="/todos", tags=["待办事项"])

@router.get("", response_model=List[ToDoResponse])
def get_all_todos(db: Session = Depends(get_db)):
    return db.query(DBToDo).all()

@router.post("", response_model=ToDoResponse)
def create_todo(todo: ToDoCreate, db: Session = Depends(get_db)):
    db_todo = DBToDo(title=todo.title, completed=todo.completed)
    db.add(db_todo)
    db.commit()
    db.refresh(db_todo)
    return db_todo

@router.delete("/{todo_id}")
def delete_todo(todo_id: int, db: Session = Depends(get_db)):
    todo = db.query(DBToDo).filter(DBToDo.id == todo_id).first()
    if not todo:
        raise HTTPException(status_code=404, detail="找不到该事项")
    db.delete(todo)
    db.commit()
    return {"message": "删除成功"}