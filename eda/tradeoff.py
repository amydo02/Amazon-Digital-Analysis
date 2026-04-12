"""
Detects reviews containing contrastive language ("but", "however", etc.)
and examines whether trade-off reviewers rate differently, what they
sacrifice, and how trade-offs vary by price tier and category.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer

CONTRAST_WORDS = [
    "but", "however", "although", "though", "despite",
    "unfortunately", "except", "yet", "still", "whereas",
]

# Aspect keywords — what customers commonly trade off
ASPECT_KEYWORDS = {
    "battery":     r"\bbatter(?:y|ies|life)\b",
    "performance": r"\b(?:fast|slow|speed|performance|lag|laggy|responsive|processor|cpu|ram)\b",
    "display":     r"\b(?:screen|display|resolution|brightness|color|colours?)\b",
    "build":       r"\b(?:build|quality|plastic|metal|feel|solid|flimsy|cheap|premium)\b",
    "keyboard":    r"\b(?:keyboard|keys|typing|trackpad|touchpad)\b",
    "price/value": r"\b(?:price|value|worth|expensive|cheap|affordable|cost)\b",
    "camera":      r"\b(?:camera|webcam|photo|video|image quality)\b",
    "software":    r"\b(?:software|bloatware|windows|os|app|driver|update)\b",
}


# ── Chart 1: Trade-off rate by category and price tier ───────────────────────

def tradeoff_rate(df: pd.DataFrame) -> go.Figure:
    """Grouped bar — % of reviews with trade-off language, by category."""
    stats = (
        df.groupby("category")["has_tradeoff"]
        .agg(["sum", "count"])
        .assign(pct=lambda x: (x["sum"] / x["count"] * 100).round(1))
        .reset_index()
        .rename(columns={"sum": "tradeoff_count", "count": "total"})
    )

    fig = px.bar(
        stats, x="category", y="pct",
        color="category",
        text=stats["pct"].astype(str) + "%",
        title="% of Reviews Containing Trade-off Language by Category",
        labels={"category": "Device Category", "pct": "% Trade-off Reviews"},
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=400, yaxis_range=[0, 60])
    return fig


# ── Chart 2: Do trade-off reviewers rate differently? ────────────────────────

def tradeoff_vs_rating(df: pd.DataFrame) -> go.Figure:
    """Side-by-side violin — rating distribution for tradeoff vs non-tradeoff."""
    df2 = df.copy()
    df2["review_type"] = df2["has_tradeoff"].map(
        {1: "Trade-off review\n('but', 'however'…)", 0: "Standard review"}
    )

    fig = px.violin(
        df2, x="review_type", y="rating",
        color="review_type",
        box=True, points=False,
        color_discrete_map={
            "Trade-off review\n('but', 'however'…)": "#e67e22",
            "Standard review": "#3498db",
        },
        title="Star Rating Distribution: Trade-off vs Standard Reviews",
        labels={"review_type": "", "rating": "Star Rating"},
    )
    fig.update_layout(showlegend=False, height=430)
    return fig


# ── Chart 3: Trade-off rate by price tier ────────────────────────────────────

def tradeoff_by_price_tier(df: pd.DataFrame) -> go.Figure:
    """Line chart — are pricier products reviewed with more trade-off language?"""
    tier_order = ["Low", "Medium", "High", "Premium"]
    with_price = df[df["price_missing"] == 0]

    stats = (
        with_price.groupby("price_tier")["has_tradeoff"]
        .agg(["sum", "count"])
        .assign(pct=lambda x: (x["sum"] / x["count"] * 100).round(1))
        .reindex(tier_order)
        .reset_index()
    )

    fig = px.line(
        stats, x="price_tier", y="pct",
        markers=True,
        title="Trade-off Language Rate by Price Tier",
        labels={"price_tier": "Price Tier", "pct": "% Trade-off Reviews"},
        color_discrete_sequence=["#9b59b6"],
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=380, yaxis_range=[0, 60])
    return fig


# ── Chart 4: What aspects appear in trade-off reviews? ───────────────────────

def tradeoff_aspect_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap — for trade-off reviews, which product aspects are mentioned
    most often, broken down by star rating group.
    """
    tradeoff_df = df[df["has_tradeoff"] == 1].copy()
    tradeoff_df["rating_group"] = pd.cut(
        tradeoff_df["rating"],
        bins=[0, 2, 3, 5],
        labels=["Very Low (1–2★)", "Low (3★)", "High (4–5★)"],
    )

    records = []
    for aspect, pattern in ASPECT_KEYWORDS.items():
        tradeoff_df[f"_asp_{aspect}"] = (
            tradeoff_df["review_text"].astype(str)
            .str.contains(pattern, case=False, regex=True)
        )
        for group in tradeoff_df["rating_group"].dropna().unique():
            subset = tradeoff_df[tradeoff_df["rating_group"] == group]
            pct = subset[f"_asp_{aspect}"].mean() * 100
            records.append({"aspect": aspect, "rating_group": str(group), "pct": round(pct, 1)})

    heat_df = pd.DataFrame(records)
    pivot   = heat_df.pivot(index="aspect", columns="rating_group", values="pct")
    col_order = ["Very Low (1–2★)", "Low (3★)", "High (4–5★)"]
    pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        text=np.round(pivot.values, 1),
        texttemplate="%{text}%",
        colorscale="YlOrRd",
        colorbar=dict(title="% of reviews"),
    ))
    fig.update_layout(
        title="Aspects Mentioned in Trade-off Reviews by Rating Group",
        height=420,
        margin=dict(l=120, b=60, t=60, r=10),
    )
    return fig


# ── Chart 5: Most common contrast sentence pairs ─────────────────────────────

def contrast_word_frequency(df: pd.DataFrame) -> go.Figure:
    """Bar chart — which contrast words appear most in reviews."""
    import re
    results = []
    for word in CONTRAST_WORDS:
        pattern = rf"\b{word}\b"
        count = df["review_text"].astype(str).str.contains(
            pattern, case=False, regex=True
        ).sum()
        results.append({"word": word, "count": int(count)})

    freq_df = pd.DataFrame(results).sort_values("count", ascending=True)

    fig = px.bar(
        freq_df, x="count", y="word",
        orientation="h",
        title="Frequency of Contrast Words Across All Reviews",
        labels={"count": "Number of Reviews", "word": "Contrast Word"},
        color="count",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=100),
    )
    return fig
