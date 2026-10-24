from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Use postgresql+psycopg://user:pass@host:5432/db in production; SQLite works for local dev.
    database_url: str = f"sqlite:///{BASE_DIR / 'corn_leaf.db'}"
    secret_key: str = "change-me-in-production"
    access_token_minutes: int = 60 * 24
    cors_origins: str = "http://localhost:3000"

    model_path: Path = BASE_DIR / "model" / "corn_leaf_disease_model_v3.keras"
    class_names_path: Path = BASE_DIR / "model" / "class_names.json"
    upload_dir: Path = BASE_DIR / "uploads"

    # Rejection of non-corn photos (any one triggers "not a corn leaf"):
    #  1) model predicts Not_Leaf (only if the model was trained with that class)
    #  2) feature-space distance above the fitted threshold (needs model/ood_stats.npz from training pipeline)
    #  3) top-class confidence below this floor
    ood_stats_path: Path = BASE_DIR / "model" / "ood_stats.npz"
    ood_scale: float = 1.0  # >1 makes the unfamiliar-image check more lenient, <1 stricter
    reject_below_confidence: float = 0.55

    max_upload_mb: int = 8
    # Valid corn results below this confidence still get a "model isn't sure" warning.
    low_confidence_threshold: float = 0.70


settings = Settings()
