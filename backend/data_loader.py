# ==============================================================
# FILE: titanic_agent/backend/data_loader.py
# PURPOSE: Fetch, validate, clean, and cache the Titanic dataset.
# ==============================================================

import logging
from functools import lru_cache
from io import StringIO

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: set[str] = {
    "PassengerId", "Survived", "Pclass", "Name",
    "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
    "Cabin", "Embarked",
}

PORT_MAP: dict[str, str] = {
    "C": "Cherbourg",
    "Q": "Queenstown",
    "S": "Southampton",
}


class DatasetValidationError(Exception):
    """Raised when the loaded dataset does not satisfy schema requirements."""


# --------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------

def _fetch_raw() -> pd.DataFrame:
    """Load CSV from a local path (if configured) or download from URL."""
    if settings.TITANIC_CSV_PATH:
        logger.info("Loading dataset from local path: %s", settings.TITANIC_CSV_PATH)
        return pd.read_csv(settings.TITANIC_CSV_PATH)

    logger.info("Downloading dataset from: %s", settings.TITANIC_CSV_URL)
    response = requests.get(settings.TITANIC_CSV_URL, timeout=15)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def _validate(df: pd.DataFrame) -> None:
    """Raise DatasetValidationError for schema violations or empty data."""
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise DatasetValidationError(
            f"Dataset is missing required columns: {sorted(missing_cols)}"
        )
    if df.empty:
        raise DatasetValidationError("Dataset loaded but is empty.")
    logger.info("Dataset validated: %d rows, %d columns.", *df.shape)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal, analysis-safe cleaning:
    - Age NaN values are preserved (not imputed) so statistical queries remain accurate.
    - Embarked NaN (2 rows) filled with mode.
    - Fare NaN filled with median.
    - Derived columns added for richer querying.
    """
    # Embarked: fill 2 missing entries with mode
    if df["Embarked"].isna().any():
        mode_val: str = df["Embarked"].mode()[0]
        df["Embarked"] = df["Embarked"].fillna(mode_val)
        logger.debug("Filled missing Embarked with mode='%s'.", mode_val)

    # Fare: fill rare NaN with median
    if df["Fare"].isna().any():
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Derived: embarkation port full name
    df["EmbarkPort"] = df["Embarked"].map(PORT_MAP)

    # Derived: age group bands
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "Young Adult", "Adult", "Senior"],
    )

    # Derived: family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    logger.info("Dataset cleaning complete.")
    return df


# --------------------------------------------------------------
# Public API
# --------------------------------------------------------------

@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """
    Return the fully validated and cleaned Titanic DataFrame.
    Result is cached in-process after the first call.
    """
    df = _fetch_raw()
    _validate(df)
    df = _clean(df)
    return df
