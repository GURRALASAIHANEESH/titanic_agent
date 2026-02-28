# ==============================================================
# FILE: titanic_agent/backend/tools.py
# PURPOSE: All LangChain-compatible tools — statistics and plots.
# ==============================================================

import base64
import io
import logging
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend required inside a server process
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain.tools import tool

from data_loader import get_dataframe

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# --------------------------------------------------------------
# Module-level chart buffer
# Shared across a single request lifecycle; reset before each run.
# --------------------------------------------------------------

_last_chart_b64: Optional[str] = None


def get_last_chart_b64() -> Optional[str]:
    return _last_chart_b64


def reset_chart_buffer() -> None:
    global _last_chart_b64
    _last_chart_b64 = None


# --------------------------------------------------------------
# Internal rendering helper
# --------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    """Serialize a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ==============================================================
# STATISTICS TOOLS
# ==============================================================

@tool
def get_dataset_summary(_: str = "") -> str:
    """
    Return a high-level summary of the Titanic dataset:
    shape, column types, missing value counts, and descriptive statistics.
    Input is ignored.
    """
    df = get_dataframe()
    sections = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "",
        "Column dtypes:",
        df.dtypes.to_string(),
        "",
        "Missing values per column:",
        df.isnull().sum().to_string(),
        "",
        "Descriptive statistics (numeric columns):",
        df.describe().round(4).to_string(),
    ]
    return "\n".join(sections)


@tool
def get_survival_stats(_: str = "") -> str:
    """
    Return survival statistics: overall rate, breakdown by gender,
    breakdown by passenger class, and breakdown by age group.
    Input is ignored.
    """
    df = get_dataframe()
    total = len(df)
    survived = int(df["Survived"].sum())
    overall_pct = survived / total * 100

    by_sex = (
        df.groupby("Sex")["Survived"]
        .agg(count="sum", rate="mean")
        .assign(rate_pct=lambda x: (x["rate"] * 100).round(2))
    )

    by_class = (
        df.groupby("Pclass")["Survived"]
        .agg(count="sum", rate="mean")
        .assign(rate_pct=lambda x: (x["rate"] * 100).round(2))
    )

    by_age_group = (
        df.groupby("AgeGroup", observed=True)["Survived"]
        .agg(count="sum", rate="mean")
        .assign(rate_pct=lambda x: (x["rate"] * 100).round(2))
    )

    return "\n".join([
        f"Overall: {survived} of {total} survived ({overall_pct:.2f}%)",
        "",
        "Survival by Gender:",
        by_sex[["count", "rate_pct"]].to_string(),
        "",
        "Survival by Passenger Class:",
        by_class[["count", "rate_pct"]].to_string(),
        "",
        "Survival by Age Group:",
        by_age_group[["count", "rate_pct"]].to_string(),
    ])


@tool
def get_gender_stats(_: str = "") -> str:
    """
    Return counts and percentages of male vs female passengers.
    Input is ignored.
    """
    df = get_dataframe()
    counts = df["Sex"].value_counts()
    pcts = (counts / len(df) * 100).round(2)
    result = pd.DataFrame({"count": counts, "percentage_%": pcts})
    return result.to_string()


@tool
def get_fare_stats(_: str = "") -> str:
    """
    Return descriptive statistics for ticket fares, including
    average fare broken down by passenger class.
    Input is ignored.
    """
    df = get_dataframe()
    overall = df["Fare"].describe().round(4)
    by_class = df.groupby("Pclass")["Fare"].mean().round(2).rename("avg_fare")
    return "\n".join([
        "Overall fare statistics:",
        overall.to_string(),
        "",
        "Average fare by passenger class:",
        by_class.to_string(),
    ])


@tool
def get_embarkation_stats(_: str = "") -> str:
    """
    Return passenger counts and percentages per embarkation port.
    Ports: C = Cherbourg, Q = Queenstown, S = Southampton.
    Input is ignored.
    """
    df = get_dataframe()
    counts = df["EmbarkPort"].value_counts()
    pcts = (counts / len(df) * 100).round(2)
    result = pd.DataFrame({"count": counts, "percentage_%": pcts})
    return result.to_string()


@tool
def get_age_stats(_: str = "") -> str:
    """
    Return age statistics: descriptive summary and age group distribution.
    Input is ignored.
    """
    df = get_dataframe()
    valid_ages = df["Age"].dropna()
    group_dist = df["AgeGroup"].value_counts().sort_index()
    return "\n".join([
        f"Age statistics (non-null count: {len(valid_ages)} of {len(df)}):",
        valid_ages.describe().round(2).to_string(),
        "",
        "Age group distribution:",
        group_dist.to_string(),
    ])


@tool
def get_family_stats(_: str = "") -> str:
    """
    Return family size distribution and survival rate for passengers
    traveling alone versus with family.
    Input is ignored.
    """
    df = get_dataframe()
    size_dist = df["FamilySize"].value_counts().sort_index().rename("passenger_count")
    alone_surv = (
        df.groupby("IsAlone")["Survived"]
        .mean()
        .round(4)
        .rename(index={0: "With Family", 1: "Alone"})
    )
    return "\n".join([
        "Family size distribution:",
        size_dist.to_string(),
        "",
        "Survival rate (alone vs with family):",
        alone_surv.to_string(),
    ])


@tool
def run_dataframe_query(expression: str) -> str:
    """
    Execute a safe, read-only pandas expression against the Titanic DataFrame.
    The DataFrame is available as the variable `df`.
    Only read operations are permitted.

    Valid examples:
      df['Age'].mean()
      df[df['Survived'] == 1]['Sex'].value_counts()
      df.groupby('Pclass')['Fare'].median()

    Destructive operations (drop, write, exec, os, sys, etc.) are blocked.
    """
    BLOCKED_PATTERN = re.compile(
        r"\b(import\s+os|import\s+sys|open\s*\(|exec\s*\(|eval\s*\(|"
        r"__import__|subprocess|shutil|rmdir|remove|unlink|write|"
        r"\.drop\s*\(|to_csv|to_sql|to_excel|to_parquet)\b",
        re.IGNORECASE,
    )
    if BLOCKED_PATTERN.search(expression):
        return (
            "Blocked: only read-only pandas expressions are permitted. "
            "Destructive or system operations are not allowed."
        )
    try:
        df = get_dataframe()  # noqa: F841 — referenced inside eval
        result = eval(  # noqa: S307
            expression,
            {"df": df, "pd": pd, "__builtins__": {}},
        )
        return str(result)
    except Exception as exc:
        logger.warning("run_dataframe_query failed for expression=%r: %s", expression, exc)
        return f"Query error: {exc}"


# ==============================================================
# PLOTTING TOOLS
# ==============================================================

@tool
def plot_histogram(column: str) -> str:
    """
    Generate a histogram for a numeric column in the Titanic dataset.
    Input: exact column name, e.g. 'Age' or 'Fare'.
    Saves the chart internally; returns a confirmation string.
    """
    global _last_chart_b64
    df = get_dataframe()
    col = column.strip()

    if col not in df.columns:
        return f"Column '{col}' not found. Available columns: {list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[col]):
        return f"Column '{col}' is not numeric and cannot be plotted as a histogram."

    data = df[col].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data.values, bins=30, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Distribution of {col}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.text(
        0.98, 0.96,
        f"n = {len(data):,}  |  mean = {data.mean():.2f}  |  median = {data.median():.2f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#555555",
    )
    fig.tight_layout()
    _last_chart_b64 = _fig_to_b64(fig)
    return f"Histogram for '{col}' generated successfully."


@tool
def plot_bar_chart(column: str) -> str:
    """
    Generate a bar chart showing value counts for a categorical column.
    Input: exact column name, e.g. 'Embarked', 'Pclass', 'Sex', 'Survived'.
    Saves the chart internally; returns a confirmation string.
    """
    global _last_chart_b64
    df = get_dataframe()
    col = column.strip()

    if col not in df.columns:
        return f"Column '{col}' not found. Available columns: {list(df.columns)}"

    counts = df[col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Passenger Count by {col}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    _last_chart_b64 = _fig_to_b64(fig)
    return f"Bar chart for '{col}' generated successfully."


@tool
def plot_survival_by_category(column: str) -> str:
    """
    Generate a grouped bar chart showing survived vs did-not-survive counts
    for each category in the given column.
    Input: categorical column name, e.g. 'Sex', 'Pclass', 'Embarked'.
    Saves the chart internally; returns a confirmation string.
    """
    global _last_chart_b64
    df = get_dataframe()
    col = column.strip()

    if col not in df.columns:
        return f"Column '{col}' not found. Available columns: {list(df.columns)}"

    pivot = (
        df.groupby([col, "Survived"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Did Not Survive", 1: "Survived"})
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Survival Outcome by {col}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.7)
    plt.xticks(rotation=0)
    fig.tight_layout()
    _last_chart_b64 = _fig_to_b64(fig)
    return f"Survival bar chart for '{col}' generated successfully."


@tool
def plot_correlation_heatmap(_: str = "") -> str:
    """
    Generate a correlation heatmap for all numeric columns in the dataset.
    Input is ignored.
    Saves the chart internally; returns a confirmation string.
    """
    global _last_chart_b64
    df = get_dataframe()
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.4, ax=ax, annot_kws={"size": 9},
        vmin=-1, vmax=1,
    )
    ax.set_title(
        "Feature Correlation Heatmap — Titanic Dataset",
        fontsize=13, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    _last_chart_b64 = _fig_to_b64(fig)
    return "Correlation heatmap generated successfully."


@tool
def plot_age_by_class(_: str = "") -> str:
    """
    Generate a box plot showing passenger age distribution per passenger class.
    Input is ignored.
    Saves the chart internally; returns a confirmation string.
    """
    global _last_chart_b64
    df = get_dataframe()
    classes = sorted(df["Pclass"].unique())
    data_by_class = [df[df["Pclass"] == c]["Age"].dropna().values for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_by_class, labels=[f"Class {c}" for c in classes], patch_artist=True)
    ax.set_title(
        "Age Distribution by Passenger Class",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Passenger Class", fontsize=12)
    ax.set_ylabel("Age", fontsize=12)
    fig.tight_layout()
    _last_chart_b64 = _fig_to_b64(fig)
    return "Box plot of age by passenger class generated successfully."


# ==============================================================
# Tool registry — imported by agent.py
# ==============================================================

ALL_TOOLS = [
    get_dataset_summary,
    get_survival_stats,
    get_gender_stats,
    get_fare_stats,
    get_embarkation_stats,
    get_age_stats,
    get_family_stats,
    run_dataframe_query,
    plot_histogram,
    plot_bar_chart,
    plot_survival_by_category,
    plot_correlation_heatmap,
    plot_age_by_class,
]
