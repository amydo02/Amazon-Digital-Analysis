"""
Goes deeper than basic price tiers to find the specific price points where
customer satisfaction meaningfully shifts.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _with_price(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["price_missing"] == 0) & (df["price"] > 0)].copy()


# ── Chart 1: Average rating by $50 price bin (the "breakpoint" chart) ────────

def rating_by_price_bin(df: pd.DataFrame, bin_size: int = 50) -> go.Figure:
    """
    Line chart — average rating across $50 price bins.
    The flattening zone = diminishing returns region.
    """
    wp = _with_price(df)
    max_price = min(wp["price"].quantile(0.97), 3000)  # cap outliers
    bins  = list(range(0, int(max_price) + bin_size, bin_size))
    wp["price_bin_mid"] = pd.cut(wp["price"], bins=bins).apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )

    binned = (
        wp.groupby("price_bin_mid", observed=True)["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"price_bin_mid": "price", "mean": "avg_rating"})
    )
    # Only show bins with enough reviews for a stable estimate
    binned = binned[binned["count"] >= 20]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=binned["price"], y=binned["avg_rating"],
        mode="lines+markers",
        line=dict(color="#3498db", width=2),
        marker=dict(size=6),
        name="Avg Rating",
        hovertemplate="$%{x:.0f} → %{y:.2f}★<extra></extra>",
    ))

    # Shade the "diminishing returns" zone — where the curve flattens
    fig.add_hrect(
        y0=4.0, y1=4.5,
        fillcolor="#f39c12", opacity=0.08,
        annotation_text="Typical satisfaction plateau",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Average Star Rating by Price (${bin_size} bins) — Breakpoint View",
        xaxis_title="Price ($)",
        yaxis_title="Average Rating",
        yaxis_range=[3.0, 5.1],
        height=450,
        hovermode="x unified",
    )
    return fig


# ── Chart 2: Review volume vs price ──────────────────────────────────────────

def volume_by_price_bin(df: pd.DataFrame, bin_size: int = 50) -> go.Figure:
    """Bar chart — how many reviews fall in each $50 price bucket."""
    wp        = _with_price(df)
    max_price = min(wp["price"].quantile(0.97), 3000)
    bins      = list(range(0, int(max_price) + bin_size, bin_size))
    wp["price_bin_mid"] = pd.cut(wp["price"], bins=bins).apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    counts = (
        wp.groupby("price_bin_mid", observed=True).size()
        .reset_index(name="reviews")
        .rename(columns={"price_bin_mid": "price"})
    )

    fig = px.bar(
        counts, x="price", y="reviews",
        title="Review Volume by Price Bucket — Where Most Buyers Shop",
        labels={"price": "Price ($)", "reviews": "Number of Reviews"},
        color="reviews",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False, height=380)
    return fig


# ── Chart 3: Best "value" products — high rating, high review count ───────────

def value_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Scatter — each bubble = one product.
    X = price, Y = average rating, size = number of reviews.
    Top-right + large bubbles = best value candidates.
    """
    wp = _with_price(df)
    products = (
        wp.groupby("parent_asin")
        .agg(
            price=("price", "median"),
            avg_rating=("rating", "mean"),
            review_count=("rating", "count"),
            product_title=("product_title", "first"),
            category=("category", "first"),
        )
        .reset_index()
    )
    # Cap outlier prices for readability
    products = products[products["price"] <= products["price"].quantile(0.97)]
    products = products[products["review_count"] >= 10]

    fig = px.scatter(
        products,
        x="price", y="avg_rating",
        size="review_count",
        color="category",
        hover_name="product_title",
        hover_data={"price": ":.2f", "avg_rating": ":.2f",
                    "review_count": True, "category": False},
        title="Product Value Map — Price vs Rating (bubble = review count)",
        labels={"price": "Median Price ($)", "avg_rating": "Avg Rating",
                "review_count": "Reviews", "category": "Category"},
        color_discrete_sequence=px.colors.qualitative.Set2,
        size_max=40,
        opacity=0.7,
    )
    fig.update_layout(height=520, yaxis_range=[1, 5.2])
    return fig


# ── Chart 4: Sentiment vs price by category ───────────────────────────────────

def sentiment_vs_price(df: pd.DataFrame) -> go.Figure:
    """
    One subplot per category — scatter of price vs Vader sentiment,
    with a LOWESS-free trend line using numpy polyfit.
    """
    wp   = _with_price(df)
    cats = sorted(wp["category"].unique())
    n    = len(cats)
    fig  = make_subplots(
        rows=1, cols=n,
        subplot_titles=[c.capitalize() for c in cats],
        shared_yaxes=True,
    )
    colors = px.colors.qualitative.Pastel

    for i, cat in enumerate(cats, 1):
        sub = wp[wp["category"] == cat].sample(
            min(5000, len(wp[wp["category"] == cat])), random_state=42
        )
        fig.add_trace(
            go.Scatter(
                x=sub["price"], y=sub["vader_sentiment"],
                mode="markers",
                marker=dict(opacity=0.3, size=4, color=colors[i % len(colors)]),
                name=cat, showlegend=False,
            ),
            row=1, col=i,
        )
        # Trend line via numpy (no statsmodels)
        try:
            coeffs = np.polyfit(sub["price"], sub["vader_sentiment"], deg=1)
            x_line = np.linspace(sub["price"].min(), sub["price"].max(), 100)
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(
                go.Scatter(
                    x=x_line, y=y_line,
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                ),
                row=1, col=i,
            )
        except Exception:
            pass
        fig.update_xaxes(title_text="Price ($)", row=1, col=i)

    fig.update_yaxes(title_text="Vader Sentiment", row=1, col=1)
    fig.update_layout(
        title="Sentiment vs Price by Device Category (with linear trend)",
        height=440,
    )
    return fig


# ── Chart 5: Negative review rate by price bin ────────────────────────────────

def negative_rate_by_price(df: pd.DataFrame, bin_size: int = 100) -> go.Figure:
    """
    Line chart — % of reviews that are 1-3 stars, across price bins.
    Reveals whether cheap or expensive products get more complaints.
    """
    wp        = _with_price(df)
    max_price = min(wp["price"].quantile(0.97), 3000)
    bins      = list(range(0, int(max_price) + bin_size, bin_size))
    wp["price_bin_mid"] = pd.cut(wp["price"], bins=bins).apply(
        lambda x: x.mid if pd.notna(x) else np.nan
    )
    wp["is_negative"] = (wp["rating"] <= 3).astype(int)

    binned = (
        wp.groupby("price_bin_mid", observed=True)["is_negative"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"price_bin_mid": "price", "mean": "neg_rate"})
    )
    binned = binned[binned["count"] >= 20]
    binned["neg_pct"] = (binned["neg_rate"] * 100).round(1)

    fig = px.line(
        binned, x="price", y="neg_pct",
        markers=True,
        title=f"% Negative Reviews (1–3★) by Price (${bin_size} bins)",
        labels={"price": "Price ($)", "neg_pct": "% Negative Reviews"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig.update_layout(height=400, yaxis_range=[0, 50])
    return fig
