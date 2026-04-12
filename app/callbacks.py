import os

import pandas as pd
import plotly.io as pio
from dash import Input, Output, dcc, html

from app.layout import (
    body_text, card, insight_note, one_col, props_table,
    research_box, section_heading, sub_section_heading, two_col,
)
from eda import category, correlation, covid, overview, price, price_breakpoints, ratings, text, time as eda_time, tradeoff

# Data source config
LOCAL_OUTPUT_CSV  = os.path.join("dataset", "eda_ready.csv")
FIGURE_CACHE_DIR  = os.path.join("dataset", "fig_cache")
USE_LOCAL_CSV = True

_df_cache: dict = {}
_fig_cache: dict = {}   # in-memory cache — survives tab switches within a session


def _load_data(force: bool = False) -> pd.DataFrame:
    if "df" not in _df_cache or force:
        if not os.path.exists(LOCAL_OUTPUT_CSV):
            raise FileNotFoundError(f"{LOCAL_OUTPUT_CSV} not found")

        print(f"[LOCAL] Loading from {LOCAL_OUTPUT_CSV} …")
        _df_cache["df"] = pd.read_csv(LOCAL_OUTPUT_CSV, low_memory=False)

        print(f"Loaded {len(_df_cache['df']):,} rows")

    return _df_cache["df"]


def _fig(key: str, fn, df):
    """Return a cached figure — checks disk first, then computes once and saves."""
    if key in _fig_cache:
        return _fig_cache[key]

    cache_path = os.path.join(FIGURE_CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            fig = pio.from_json(f.read())
        _fig_cache[key] = fig
        return fig

    os.makedirs(FIGURE_CACHE_DIR, exist_ok=True)
    fig = fn(df)
    with open(cache_path, "w") as f:
        f.write(pio.to_json(fig))
    _fig_cache[key] = fig
    return fig


# Analytics sub-tab definitions
_ANALYTICS_SUB_TABS = [
    ("sub-ratings",    "RATINGS & PRICE"),
    ("sub-time-text",  "TIME & TEXT"),
    ("sub-category",   "CATEGORY"),
    ("sub-price-bp",   "PRICE BREAKPOINTS"),
    ("sub-covid",      "COVID TRENDS"),
    ("sub-tradeoff",   "TRADE-OFFS"),
]


# Register callbacks

def register_callbacks(app):

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab_content(tab):
        if tab == "tab-overview":
            return _render_overview_page()
        if tab == "tab-methods":
            return _render_methods_page()

        # Data-dependent tabs
        try:
            df = _load_data()
        except Exception as e:
            return html.Div(
                f"Could not load data: {e}",
                style={"color": "red", "padding": "20px",
                       "whiteSpace": "pre-wrap",
                       "fontFamily": "monospace"},
            )

        if tab == "tab-dataset":
            return _render_dataset_page(df)
        if tab == "tab-analytics":
            return _render_analytics_shell()

        return html.Div("Unknown tab.")

    @app.callback(
        Output("analytics-sub-content", "children"),
        Input("analytics-sub-tabs", "value"),
    )
    def render_analytics_sub(sub_tab):
        try:
            df = _load_data()
        except Exception as e:
            return html.Div(f"Could not load data: {e}",
                            style={"color": "red", "fontFamily": "monospace"})

        renderers = {
            "sub-ratings":    _render_sub_ratings,
            "sub-time-text":  _render_sub_time_text,
            "sub-category":   _render_sub_category,
            "sub-price-bp":   _render_sub_price_breakpoints,
            "sub-covid":      _render_sub_covid,
            "sub-tradeoff":   _render_sub_tradeoff,
        }
        fn = renderers.get(sub_tab)
        if fn is None:
            return html.Div("Unknown section.")
        return fn(df)


# Tab page renderers

def _render_overview_page():
    return html.Div([
        card(
            section_heading("Project Overview"),
            body_text(
                "Amazon is one of the world's largest marketplaces for digital devices. "
                "With millions of customer reviews spanning laptops, tablets, and desktops, "
                "this dataset offers a rich opportunity to understand how consumers evaluate "
                "technology products - from star ratings and pricing patterns to the language "
                "they use when expressing satisfaction or disappointment."
            ),
            body_text(
                "This project collects and cleans Amazon product review data, applies a "
                "machine-learning classifier to filter for genuine digital device reviews, "
                "and then performs a comprehensive exploratory data analysis (EDA). "
                "We examine rating distributions, price-tier effects, seasonal trends, "
                "review text length, VADER sentiment scores, brand performance, and "
                "feature correlations."
            ),
            # for my teammate: you can add more questioons here if you wants
            research_box(
                html.H4("Research Question",
                        style={"color": "#1E2736", "margin": "0 0 8px",
                               "fontFamily": "Segoe UI, Arial, sans-serif",
                               "fontSize": "0.95rem"}),
                html.Ul(
                    [
                        html.Li("What factors do device category, price tier, brand, or review sentiment, best predict whether a customer leaves a high (≥ 4★) or low (≤ 2★) rating?"),
                        html.Li("How have review volumes and spending patterns shifted over time?"),
                    ],
                    style={"color": "#2d2d2d", "fontSize": "0.95rem",
                           "fontFamily": "Segoe UI, Arial, sans-serif",
                           "lineHeight": "1.75", "margin": "0", "paddingLeft": "20px"},
                ),
            ),
        ),
        card(
            section_heading("Key Highlights"),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)",
                       "gap": "20px"},
                children=[
                    _stat_card("600K+",  "Reviews collected"),
                    _stat_card("3",      "Device categories\n(Laptop · Tablet · Desktop)"),
                    _stat_card("2018 - 2023", "Review time span"),
                ],
            ),
        ),
    ])


def _render_dataset_page(df: pd.DataFrame):
    n_rows    = f"{len(df):,}"
    n_brands  = f"{df['brand'].nunique():,}" if "brand" in df.columns else "—"
    n_products = f"{df['parent_asin'].nunique():,}" if "parent_asin" in df.columns else "—"
    years     = f"{int(df['year'].min())} - {int(df['year'].max())}" if "year" in df.columns else "—"

    return html.Div([
        card(
            section_heading("Dataset"),
            body_text(
                "Dataset is open source from Hugging Face: https://amazon-reviews-2023.github.io. "
                "Each row represents one customer review "
                "and is linked to its product's metadata - including listed price, brand, "
                "and category. Reviews are filtered with an ML classifier and further "
                "enriched with computed features: price tier, VADER sentiment score, "
                "review language, and temporal attributes."
            ),
            html.Div(style={"marginTop": "20px"},
                     children=[props_table([
                         ("Total reviews (EDA-ready)",   n_rows),
                         ("Unique products",              n_products),
                         ("Unique brands",                n_brands),
                         ("Review period",                years),
                         ("Device categories",            "Laptop, Tablet, Desktop"),
                         ("Rating scale",                 "1 - 5 stars"),
                         ("Source",                       "Amazon Product Reviews (public)"),
                         ("Language handling",            "Non-English reviews translated via Google Translate"),
                     ])]),
        ),
        card(
            section_heading("Data Summary"),
            one_col(overview.summary_table(df)),
            one_col(overview.describe_table(df)),
        ),
    ])


def _render_methods_page():
    def step_card(num, color, title, description):
        return html.Div(
            style={
                "display": "flex", "gap": "20px", "alignItems": "flex-start",
                "backgroundColor": CARD_BG, "borderRadius": "6px",
                "padding": "24px", "marginBottom": "16px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                "borderLeft": f"5px solid {color}",
            },
            children=[
                html.Div(str(num), style={
                    "backgroundColor": color, "color": "white",
                    "borderRadius": "50%", "width": "36px", "height": "36px",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontWeight": "700", "fontSize": "1rem", "flexShrink": "0",
                    "fontFamily": "Segoe UI, Arial, sans-serif",
                }),
                html.Div([
                    html.H4(title, style={"margin": "0 0 8px", "color": "#1E2736",
                                          "fontFamily": "Segoe UI, Arial, sans-serif",
                                          "fontSize": "1rem"}),
                    body_text(description, margin="0"),
                ]),
            ],
        )

    return html.Div([
        card(
            section_heading("Methods"),
            body_text(
                "The data pipeline consists of three sequential steps that transform "
                "raw Amazon review exports into a clean, analysis-ready table."
            ),
        ),
        step_card(1, "#7FAAC4", "Step 1 — ML Filter",
                  "A scikit-learn text classifier is trained on product titles and metadata "
                  "to distinguish genuine digital devices (laptops, tablets, desktops) from "
                  "unrelated products that may appear in the same broad category. "
                  "This step ensures downstream analyses are not polluted by accessories, "
                  "cases, or unrelated electronics."),
        step_card(2, "#1E2736", "Step 2 — BigQuery SQL",
                  "Filtered products are joined with their full review histories using "
                  "BigQuery SQL. Duplicate reviews are removed, and the reviews table is "
                  "linked to product metadata (brand, price, category). "
                  "The output is a single denormalized table ready for EDA."),
        step_card(3, "#34527A", "Step 3 — EDA Data Preparation",
                  "The merged table is cleaned and enriched: prices are bucketed into "
                  "Low / Medium / High / Premium tiers using quantiles; VADER sentiment "
                  "scores are computed for each review text; non-English reviews are "
                  "detected with Lingua and translated via Google Translate; "
                  "temporal features (year, month, day-of-week) are extracted; "
                  "and a price_missing flag is added for out-of-stock products. "
                  "The result is saved as dataset/eda_ready.csv and uploaded to BigQuery."),
    ])


def _render_analytics_shell():
    """Renders the analytics page skeleton with sub-tabs. Content loads per sub-tab."""
    base = {
        "backgroundColor": "transparent",
        "color": "rgba(30,39,54,0.55)",
        "border": "none",
        "borderBottom": "3px solid transparent",
        "padding": "10px 18px",
        "fontFamily": "Segoe UI, Arial, sans-serif",
        "letterSpacing": "1.5px",
        "fontSize": "0.74rem",
        "fontWeight": "600",
        "display": "flex",
        "alignItems": "center",
        "justifyContent": "center",
    }
    selected = {
        **base,
        "color": "#34527A",
        "borderBottom": "3px solid #34527A",
    }

    tabs = [
        dcc.Tab(label=label, value=value, style=base, selected_style=selected)
        for value, label in _ANALYTICS_SUB_TABS
    ]

    return html.Div([
        card(
            section_heading("Major Findings"),
            body_text("We're working on this part. Will update this section soon!"),
        ),

        # Sub-tab bar — matches the feel of the main nav tabs
        html.Div(
            style={
                "backgroundColor": "#ffffff",
                "borderRadius": "6px 6px 0 0",
                "borderBottom": f"2px solid #D9EEF7",
                "marginBottom": "0",
            },
            children=[
                dcc.Tabs(
                    id="analytics-sub-tabs",
                    value="sub-ratings",
                    children=tabs,
                    style={"border": "none"},
                    colors={"border": "transparent", "primary": "transparent",
                            "background": "transparent"},
                ),
            ],
        ),

        # Content flows directly below with no gap
        dcc.Loading(
            type="default",
            color="#34527A",
            children=html.Div(id="analytics-sub-content"),
        ),
    ])


# Analytics sub-section renderers

def _render_sub_ratings(df: pd.DataFrame):
    return html.Div([
        card(
            sub_section_heading("⭐  Rating Analysis"),
            one_col(_fig("rating_distribution",   ratings.rating_distribution,   df)),
            insight_note(
                "Even though individual ratings usually track average_rating closely, "
                "there is a noticeable left tail (negative deltas) — worth investigating further."
            ),
            one_col(_fig("rating_delta_histogram", ratings.rating_delta_histogram, df)),
            one_col(_fig("popularity_vs_rating",   ratings.popularity_vs_rating,   df)),
            insight_note(
                "Products with more reviews tend to have slightly higher average ratings, "
                "but the trend is weak — popularity does not strongly predict satisfaction."
            ),
        ),
        card(
            sub_section_heading("💰  Price Analysis"),
            insight_note(
                "185K+ reviews belong to products with no listed price "
                "(out-of-stock / discontinued). A price_missing signal handles these separately."
            ),
            two_col(_fig("price_distribution",    price.price_distribution,    df),
                    _fig("rating_by_price_tier",  price.rating_by_price_tier,  df)),
            insight_note(
                "No strong differences in rating distribution across price tiers — "
                "price tier alone is not a great predictor of customer satisfaction."
            ),
            two_col(_fig("premium_price_boxplot", price.premium_price_boxplot, df),
                    _fig("spending_over_time",    price.spending_over_time,    df)),
            insight_note(
                "The Premium tier still shows extreme outliers worth investigating. "
                "Every year, December has the highest estimated spending."
            ),
        ),
    ])


def _render_sub_time_text(df: pd.DataFrame):
    return html.Div([
        card(
            sub_section_heading("📅  Time Analysis"),
            two_col(_fig("reviews_by_year",        eda_time.reviews_by_year,        df),
                    _fig("reviews_by_month",       eda_time.reviews_by_month,       df)),
            insight_note(
                "A big volume shift started in 2020 (COVID). "
                "Data only runs through Sep 2023, so that year looks low — "
                "compare 2018 - 2022 for fair year-over-year analysis. "
                "December and January consistently peak; September onward loses volume in the dataset."
            ),
            two_col(_fig("reviews_by_day_of_month", eda_time.reviews_by_day_of_month, df),
                    _fig("reviews_by_day_of_week",  eda_time.reviews_by_day_of_week,  df)),
            insight_note(
                "Fewer reviews are written on weekends. "
                "Day-of-month shows no strong pattern (lower count on day 31 is expected — "
                "not all months have 31 days)."
            ),
        ),
        card(
            sub_section_heading("📝  Text & Sentiment"),
            one_col(_fig("review_length_histogram",  text.review_length_histogram,  df)),
            insight_note(
                "Most reviews are short (< 300 characters) and the distribution is "
                "right-skewed. A small number of reviews are very long (> 2,000 characters). "
                "1-star reviews tend to be longer; 5-star reviews are shorter on average."
            ),
            two_col(_fig("review_length_by_rating", text.review_length_by_rating, df),
                    _fig("vader_by_rating",         text.vader_by_rating,         df)),
            insight_note(
                "VADER generally tracks star ratings correctly. "
                "Some overlap between 4★ and 5★ — very enthusiastic language can confuse "
                "the model, or reviewers may have given a high star rating despite negative text."
            ),
        ),
    ])


def _render_sub_category(df: pd.DataFrame):
    return html.Div([
        card(
            sub_section_heading("🖥️  Category & Brand"),
            two_col(_fig("category_distribution",    category.category_distribution,    df),
                    _fig("category_rating_boxplot",  category.category_rating_boxplot,  df)),
            two_col(_fig("top_brands_bar",           category.top_brands_bar,           df),
                    _fig("brand_avg_rating",         category.brand_avg_rating,         df)),
            insight_note(
                "Laptops dominate the review volume. "
                "Rating distributions are broadly similar across categories, "
                "with slight variation in median scores by brand."
            ),
        ),
        card(
            sub_section_heading("🔗  Feature Correlation"),
            one_col(_fig("correlation_heatmap", correlation.correlation_heatmap, df)),
            insight_note(
                "Strong correlation between rating and average_rating as expected. "
                "Price shows weak correlation with rating, confirming the price-tier analysis above."
            ),
        ),
    ])


def _render_sub_price_breakpoints(df: pd.DataFrame):
    return html.Div([
        card(
            sub_section_heading("📊  Price Breakpoints & Diminishing Returns"),
            one_col(_fig("rating_by_price_bin",      price_breakpoints.rating_by_price_bin,      df)),
            insight_note(
                "The curve typically flattens above a certain price point — paying more does not "
                "guarantee higher customer satisfaction."
            ),
            two_col(_fig("volume_by_price_bin",      price_breakpoints.volume_by_price_bin,      df),
                    _fig("negative_rate_by_price",   price_breakpoints.negative_rate_by_price,   df)),
            insight_note(
                "Most reviews cluster in the budget-to-mid range. "
                "The negative review rate often dips in the mid-price range before rising again at extremes."
            ),
            one_col(_fig("value_scatter",            price_breakpoints.value_scatter,            df)),
            insight_note(
                "Large bubbles in the top-right quadrant represent the best-value products — "
                "high ratings, many reviews, and reasonable price."
            ),
            one_col(_fig("sentiment_vs_price",       price_breakpoints.sentiment_vs_price,       df)),
        ),
    ])


def _render_sub_covid(df: pd.DataFrame):
    return html.Div([
        card(
            sub_section_heading("🦠  COVID-Era Trends"),
            two_col(_fig("volume_by_period",         covid.volume_by_period,         df),
                    _fig("rating_by_period",         covid.rating_by_period,         df)),
            insight_note(
                "Review volume surged during COVID as device demand spiked for remote work and learning. "
                "Average ratings remained relatively stable across all three periods."
            ),
            one_col(_fig("keyword_shift",            covid.keyword_shift,            df)),
            one_col(_fig("keyword_lift",             covid.keyword_lift,             df)),
            insight_note(
                "Keywords like 'webcam', 'zoom', and 'work from home' saw the biggest lift "
                "during COVID, reflecting the shift to remote work and video conferencing."
            ),
            two_col(_fig("sentiment_by_period",      covid.sentiment_by_period,      df),
                    _fig("category_shift_by_period", covid.category_shift_by_period, df)),
        ),
    ])


def _render_sub_tradeoff(df: pd.DataFrame):
    return html.Div([
        card(
            sub_section_heading("⚖️  Trade-off Language Analysis"),
            two_col(_fig("tradeoff_rate",            tradeoff.tradeoff_rate,            df),
                    _fig("tradeoff_vs_rating",       tradeoff.tradeoff_vs_rating,       df)),
            insight_note(
                "A notable share of reviews contain contrastive language ('but', 'however', etc.), "
                "suggesting customers often see both pros and cons in their purchases."
            ),
            two_col(_fig("tradeoff_by_price_tier",   tradeoff.tradeoff_by_price_tier,   df),
                    _fig("contrast_word_frequency",  tradeoff.contrast_word_frequency,  df)),
            insight_note(
                "Higher-priced products attract more trade-off language — buyers have higher "
                "expectations and are more likely to note specific compromises."
            ),
            one_col(_fig("tradeoff_aspect_heatmap",  tradeoff.tradeoff_aspect_heatmap,  df)),
            insight_note(
                "Performance and price/value are the most commonly traded-off aspects, "
                "especially in lower-rated reviews."
            ),
        ),
    ])


# Private helpers

CARD_BG = "#ffffff"


def _stat_card(value: str, label: str) -> html.Div:
    return html.Div(
        style={
            "backgroundColor": "#EBF4FA",
            "borderRadius": "6px",
            "padding": "20px 24px",
            "textAlign": "center",
            "borderTop": "3px solid #34527A",
        },
        children=[
            html.Div(value, style={"fontSize": "1.8rem", "fontWeight": "700",
                                   "color": "#34527A", "fontFamily": "Segoe UI, Arial, sans-serif",
                                   "whiteSpace": "pre-line"}),
            html.Div(label, style={"fontSize": "0.82rem", "color": "#555",
                                   "marginTop": "6px", "fontFamily": "Segoe UI, Arial, sans-serif",
                                   "whiteSpace": "pre-line"}),
        ],
    )
