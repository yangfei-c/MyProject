from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# 1. 数据库配置和模型
database_url = "sqlite:///./todos.db"
engine = create_engine(database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBToDo(Base):
    __tablename__ = "todos"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    completed = Column(Boolean, default=False)

Base.metadata.create_all(engine)

class ToDoCreate(BaseModel):
    title: str
    completed: bool = False

class ToDoResponse(BaseModel):
    id: int
    title: str
    completed: bool
    class Config:
        from_attributes = True

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. 部门经理路由
router = APIRouter(
    prefix="/todos",
    tags=["待办事项"]
)

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