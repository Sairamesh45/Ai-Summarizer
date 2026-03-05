import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from app.models.user import UserRoleEnum


# ── Shared ────────────────────────────────────────────────────────────────────
class UserBase(BaseModel):
    email: EmailStr
    full_name: str | None = None
    is_active: bool = True


# ── Request schemas ───────────────────────────────────────────────────────────
class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRoleEnum = UserRoleEnum.staff


class UserUpdate(BaseModel):
    full_name: str | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)
    role: UserRoleEnum | None = None


# ── Response schemas ──────────────────────────────────────────────────────────
class UserOut(UserBase):
    id: uuid.UUID
    role: UserRoleEnum
    is_superuser: bool
    avatar_url: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ── Token schemas ─────────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    sub: str  # user id
    role: str = "staff"
    exp: int | None = None
    type: str = "access"
