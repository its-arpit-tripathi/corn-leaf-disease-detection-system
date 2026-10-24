from datetime import datetime, timezone
from sqlalchemy import DateTime, Float, ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(500))
    heatmap_path: Mapped[str] = mapped_column(String(500))
    predicted_class: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    probabilities_json: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    user: Mapped[User] = relationship(back_populates="predictions")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
