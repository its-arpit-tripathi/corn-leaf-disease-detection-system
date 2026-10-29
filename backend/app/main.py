import json
import logging
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import timedelta, timezone

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .config import settings
from .db import Base, Prediction, User, engine, get_db, utcnow
from .disease_info import INFO, LABELS, NON_MAIZE_CLASSES, canon_class, get_disease_info
from .ml import Classifier, InvalidImageError
from .schemas import DiseaseInfo, PredictionOut, RegisterIn, StatsOut, TokenOut, UserOut
from .security import create_access_token, get_current_user, hash_password, verify_password

log = logging.getLogger("corn_leaf")
logging.basicConfig(level=logging.INFO)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.classifier = Classifier()  # load the model once at start-up
    log.info("Model loaded: classes=%s", app.state.classifier.class_names)
    yield


app = FastAPI(title="Corn Leaf Disease Detection API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_out(p: Prediction, reason: str | None = None, suspected: str | None = None) -> PredictionOut:
    probs = json.loads(p.probabilities_json)
    cls_key = canon_class(p.predicted_class)
    is_corn = cls_key not in NON_MAIZE_CLASSES

    info_dict = get_disease_info(cls_key)
    created = p.created_at if p.created_at.tzinfo else p.created_at.replace(tzinfo=timezone.utc)

    disease_name = info_dict.get("disease_name", LABELS.get(cls_key, cls_key))
    crop_name = info_dict.get("crop_name", "Maize (Corn)" if is_corn else "Non-Maize / Unrecognized")
    status_label = info_dict.get("status", "Healthy" if cls_key == "Healthy" else ("Not Maize" if not is_corn else "Diseased"))
    pathogen = info_dict.get("pathogen", "")
    severity = info_dict.get("severity", "")

    return PredictionOut(
        id=p.id,
        filename=p.filename,
        predicted_class=cls_key,
        disease_name=disease_name,
        crop_name=crop_name,
        status=status_label,
        pathogen=pathogen,
        severity=severity,
        label=LABELS.get(cls_key, cls_key),
        confidence=p.confidence,
        probabilities=probs,
        is_corn_leaf=is_corn,
        is_maize=is_corn,
        low_confidence=is_corn and p.confidence < settings.low_confidence_threshold,
        rejection_reason=reason,
        suspected_disease=suspected,
        info=DiseaseInfo(**info_dict),
        created_at=created,
    )


@app.get("/api/health")
def health():
    ready = getattr(app.state, "classifier", None) is not None
    clf = getattr(app.state, "classifier", None)
    return {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "classes": clf.class_names if clf else [],
        "ood_check": bool(clf and clf.ood),
    }


# ---------- auth ----------
@app.post("/api/auth/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    user = User(name=body.name.strip(), email=body.email.lower(), password_hash=hash_password(body.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    return TokenOut(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@app.post("/api/auth/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.lower()))
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return TokenOut(access_token=create_access_token(user.id), user=UserOut.model_validate(user))


@app.get("/api/users/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


# ---------- prediction ----------
@app.post("/api/predict", response_model=PredictionOut, status_code=201)
async def predict(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Unsupported file type. Upload a JPEG, PNG, WebP or BMP image.")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image is larger than {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "Empty file")

    try:
        result = await run_in_threadpool(app.state.classifier.predict_bytes, data)
    except InvalidImageError as e:
        raise HTTPException(422, str(e))

    reason = app.state.classifier.reject_reason(result)
    raw_class = canon_class(result.class_name)

    if reason:
        log.info(
            "Rejected as non-maize: reason=%s raw_class=%s conf=%.3f ood=%s veg=%.3f",
            reason,
            raw_class,
            result.confidence,
            result.ood_distance,
            result.vegetation_score,
        )
        saved_class = "Not_Corn"
        suspected = LABELS.get(raw_class, raw_class) if raw_class not in NON_MAIZE_CLASSES else None
    else:
        saved_class = raw_class
        suspected = None

    folder = settings.upload_dir / str(user.id)
    folder.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    img_path, heat_path = folder / f"{stem}.jpg", folder / f"{stem}_gradcam.png"
    img_path.write_bytes(result.image_png)
    heat_path.write_bytes(result.heatmap_png)

    pred = Prediction(
        user_id=user.id,
        filename=(file.filename or "upload")[:255],
        image_path=str(img_path),
        heatmap_path=str(heat_path),
        predicted_class=saved_class,
        confidence=result.confidence,
        probabilities_json=json.dumps(result.probabilities),
    )
    db.add(pred)
    db.commit()
    return to_out(pred, reason=reason, suspected=suspected)


def _owned(db: Session, user: User, pid: int) -> Prediction:
    pred = db.get(Prediction, pid)
    if pred is None or pred.user_id != user.id:
        raise HTTPException(404, "Prediction not found")
    return pred


@app.get("/api/predictions", response_model=list[PredictionOut])
def history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = db.scalars(
        select(Prediction)
        .where(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [to_out(r) for r in rows]


@app.get("/api/predictions/{pid}", response_model=PredictionOut)
def get_prediction(pid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return to_out(_owned(db, user, pid))


NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/api/predictions/{pid}/image")
def prediction_image(pid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return FileResponse(_owned(db, user, pid).image_path, media_type="image/jpeg", headers=NO_CACHE_HEADERS)


@app.get("/api/predictions/{pid}/heatmap")
def prediction_heatmap(pid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return FileResponse(_owned(db, user, pid).heatmap_path, media_type="image/png", headers=NO_CACHE_HEADERS)


@app.delete("/api/predictions/{pid}", status_code=204)
def delete_prediction(pid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from pathlib import Path

    pred = _owned(db, user, pid)
    for p in (pred.image_path, pred.heatmap_path):
        Path(p).unlink(missing_ok=True)
    db.delete(pred)
    db.commit()


@app.get("/api/stats", response_model=StatsOut)
def stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.execute(
        select(Prediction.predicted_class, Prediction.confidence, Prediction.created_at).where(
            Prediction.user_id == user.id
        )
    ).all()
    counts = Counter(canon_class(r.predicted_class) for r in rows)
    today = utcnow().date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    per_day = Counter(r.created_at.date() for r in rows)
    healthy = counts.get("Healthy", 0)
    not_leaf = counts.get("Not_Corn", 0) + counts.get("Not_Leaf", 0)
    clf = getattr(app.state, "classifier", None)
    clf_classes = clf.class_names if clf is not None else []
    all_known = dict.fromkeys([canon_class(c) for c in [*clf_classes, "Not_Corn", "Healthy", "Blight", "Common_Rust", "Gray_Leaf_Spot"]])

    return StatsOut(
        total=len(rows),
        healthy=healthy,
        diseased=len(rows) - healthy - not_leaf,
        not_leaf=not_leaf,
        by_class={n: counts.get(n, 0) for n in all_known if n},
        average_confidence=(
            (sum(scored) / len(scored))
            if (scored := [r.confidence for r in rows if canon_class(r.predicted_class) not in NON_MAIZE_CLASSES])
            else 0.0
        ),
        last_7_days=[{"date": d.isoformat(), "count": per_day.get(d, 0)} for d in days],
    )
