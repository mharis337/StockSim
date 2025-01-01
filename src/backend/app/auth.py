# auth.py

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging

from config import SETTINGS
from database import users_collection

router = APIRouter(
    prefix="/api",
    tags=["Authentication"]
)

# Logger setup
logger = logging.getLogger(__name__)

# Pydantic models
class User(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")

# Utility functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SETTINGS.SECRET_KEY, algorithm=SETTINGS.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SETTINGS.SECRET_KEY, algorithms=[SETTINGS.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=401, detail="Authentication failed")

# Routes
@router.post("/register")
async def register(user: User):
    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="User already exists")
    
    hashed_password = pwd_context.hash(user.password)
    
    users_collection.insert_one({
        "email": user.email,
        "password": hashed_password,
        "balance": 1000.0,  # Initial balance
        "holdings": {}
    })
    
    return {"message": "User registered successfully with $1000 balance and 0 stocks"}

@router.post("/login", response_model=Token)
async def login(user: User):
    try:
        user_data = users_collection.find_one({"email": user.email})
        
        if not user_data or not pwd_context.verify(user.password, user_data["password"]):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid email or password"}
            )

        access_token = create_access_token(data={"sub": user.email})
        
        response = JSONResponse(
            content={
                "access_token": access_token,
                "token_type": "bearer"
            }
        )
        
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=SETTINGS.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@router.post("/logout")
async def logout():
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie("auth_token")
    return response

@router.post("/refresh")
async def refresh_token(current_user: str = Depends(get_current_user)):
    try:
        access_token = create_access_token(
            data={"sub": current_user},
            expires_delta=timedelta(minutes=SETTINGS.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        response = JSONResponse(
            content={
                "access_token": access_token,
                "token_type": "bearer"
            }
        )
        
        response.set_cookie(
            key="auth_token",
            value=access_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=SETTINGS.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
        return response
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Could not refresh token"
        )
@router.get("/protected")
async def protected_route(current_user: str = Depends(get_current_user)):
    """
    A protected endpoint that returns information about the current user.
    Only accessible to authenticated users.
    """
    return {"message": f"Hello, {current_user}! You have accessed a protected route."}