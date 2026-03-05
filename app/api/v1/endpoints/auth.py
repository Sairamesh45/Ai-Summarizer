import uuid
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError

from app.api.deps import AuthServiceDep, CurrentUser
from app.core.exceptions import UnauthorizedException
from app.core.security import decode_token
from app.schemas.user import Token, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: UserCreate, service: AuthServiceDep):
    user = await service.register(payload)
    return user


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    service: AuthServiceDep,
):
    return await service.login(form_data.username, form_data.password)


@router.post("/refresh", response_model=Token)
async def refresh(refresh_token: str, service: AuthServiceDep):
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise UnauthorizedException("Invalid token type")
        user_id = uuid.UUID(payload["sub"])
    except (JWTError, ValueError, KeyError):
        raise UnauthorizedException("Invalid refresh token")

    return await service.refresh_tokens(user_id)


@router.get("/me", response_model=UserOut)
async def me(current_user: CurrentUser):
    return current_user
