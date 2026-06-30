from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import List
app=FastAPI(title="待办事项")

class ToDoItem(BaseModel):
    id:int
    title:str
    completed:bool

moni_db:List[ToDoItem]=[]

@app.get("/")
def read_root():
    return {"message": "Welcome to the To-Do List API"}

@app.get("/todos",response_model=List[ToDoItem])
def get_all_todos():
    return moni_db

@app.post("/todos",response_model=ToDoItem)
def create_todo(todo:ToDoItem):
    for item in moni_db:
        if item.id==todo.id:
            raise HTTPException(status_code=400,detail="ID already exists")

    moni_db.append(todo)
    return todo