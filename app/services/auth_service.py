import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConflictException,
    NotFoundException,
    UnauthorizedException,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.user import Token, UserCreate


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Helpers ───────────────────────────────────────────────────────────────
    async def _get_by_email(self, email: str) -> User | None:
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def _get_by_id(self, user_id: uuid.UUID) -> User | None:
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    # ── Public ────────────────────────────────────────────────────────────────
    async def register(self, data: UserCreate) -> User:
        existing = await self._get_by_email(data.email)
        if existing:
            raise ConflictException("Email already registered")

        user = User(
            email=data.email,
            hashed_password=hash_password(data.password),
            full_name=data.full_name,
            role=data.role,
        )
        self.db.add(user)
        await self.db.flush()  # get generated id without committing
        await self.db.refresh(user)
        return user

    async def login(self, email: str, password: str) -> Token:
        user = await self._get_by_email(email)
        if not user or not verify_password(password, user.hashed_password):
            raise UnauthorizedException("Invalid credentials")
        if not user.is_active:
            raise UnauthorizedException("Account is inactive")

        return Token(
            access_token=create_access_token(str(user.id), role=user.role.value),
            refresh_token=create_refresh_token(str(user.id)),
        )

    async def refresh_tokens(self, user_id: uuid.UUID) -> Token:
        user = await self._get_by_id(user_id)
        if not user or not user.is_active:
            raise UnauthorizedException("Account not found or inactive")

        return Token(
            access_token=create_access_token(str(user.id), role=user.role.value),
            refresh_token=create_refresh_token(str(user.id)),
        )

    async def get_user_or_404(self, user_id: uuid.UUID) -> User:
        user = await self._get_by_id(user_id)
        if not user:
            raise NotFoundException("User not found")
        return user
