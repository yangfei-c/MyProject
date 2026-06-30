from fastapi import APIRouter, Depends, HTTPException
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

pwd_content = CryptContext(schemes=["bcrypt"], deprecated="auto")

from database import get_db
from table import User

class UserCreate(BaseModel):
    username: str
    password: str



router = APIRouter(prefix="/auth", tags=["用户系统"])

@router.post("/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="该用户名已经存在")
    hashed_pwd=pwd_content.hash(user.password)
    new_user=User(username=user.username,hash_password=hashed_pwd)
    db.add(new_user)
    db.commit()

    return {"message":f"欢迎您，{user.username}，注册成功！"}

