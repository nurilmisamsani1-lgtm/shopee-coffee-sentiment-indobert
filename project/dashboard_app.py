import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from build_pipeline import IndonesianSentimentModel, MODEL_DIR, MODEL_ID, normalize_text

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_OK = True
except Exception:
    WORDCLOUD_OK = False


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data_cleaned.csv"
METRICS_PATH = PROJECT_DIR / "product_metrics.csv"
INSIGHTS_PATH = PROJECT_DIR / "insights_report.md"
MODEL_INFO_PATH = PROJECT_DIR / "sentiment_model_source.txt"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_resource
def load_sentiment_model() -> IndonesianSentimentModel:
    return IndonesianSentimentModel.load(model_id=MODEL_ID, model_dir=MODEL_DIR)


def render_wordcloud(text: str, title: str):
    st.subheader(title)
    if not WORDCLOUD_OK:
        st.info("WordCloud package belum terpasang. Jalankan: pip install wordcloud")
        return
    if not text.strip():
        st.warning("Data teks kosong untuk filter ini.")
        return

    wc = WordCloud(width=1100, height=550, background_color="white", collocations=False)
    wc.generate(text)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def sanitize_wordcloud_text(series: pd.Series) -> str:
    text = " ".join(series.fillna("").astype(str))
    text = re.sub(r"(?i)\bnan\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_ranking_view(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    perf = (
        data.groupby("product_name", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
        )
    )

    C = perf["avg_rating"].mean()
    m = max(1.0, perf["review_count"].quantile(0.6))
    v = perf["review_count"]
    R = perf["avg_rating"]
    perf["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    rating_norm = (perf["avg_rating"] - 1) / 4
    sent_norm = (perf["avg_sentiment"] + 1) / 2
    perf["csi"] = (0.6 * rating_norm + 0.4 * sent_norm) * 100

    global_rating = perf["avg_rating"].mean()
    global_sent = perf["avg_sentiment"].mean()
    perf["segment"] = "steady"
    perf.loc[(perf["avg_rating"] < global_rating) & (perf["avg_sentiment"] > global_sent), "segment"] = "undervalued"
    perf.loc[(perf["avg_rating"] > global_rating) & (perf["avg_sentiment"] < global_sent), "segment"] = "overrated"

    high_potential_mask = (
        (perf["review_count"] <= perf["review_count"].median())
        & (perf["avg_sentiment"] >= perf["avg_sentiment"].quantile(0.75))
        & (perf["avg_rating"] >= 4.0)
    )
    perf.loc[high_potential_mask, "segment"] = "high_potential"

    return perf.sort_values(["weighted_rating", "avg_sentiment"], ascending=False)


def main():
    st.set_page_config(page_title="Shopee Coffee Shop Sentiment Analysis IndoBERT", layout="wide")
    st.title("Shopee Product Review Dashboard")
    st.caption("EDA + Sentiment + Business Insights")

    if not DATA_PATH.exists():
        st.error("File data belum tersedia. Jalankan dulu: python project/build_pipeline.py")
        return

    df = load_data()

    products = sorted(df["product_name"].dropna().unique().tolist())
    sentiments = sorted(df["sentiment_label"].dropna().unique().tolist())

    st.sidebar.header("Filters")
    selected_products = st.sidebar.multiselect("Product", products, default=products[: min(10, len(products))])
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)
    rating_min, rating_max = st.sidebar.slider("Rating", 1, 5, (1, 5))

    filtered = df[
        df["product_name"].isin(selected_products)
        & df["sentiment_label"].isin(selected_sentiments)
        & df["rating"].between(rating_min, rating_max)
    ].copy()

    tab_dashboard, tab_test = st.tabs(["Dashboard", "Uji Sentimen Teks"])

    with tab_dashboard:
        if filtered.empty:
            st.warning("Tidak ada data untuk kombinasi filter saat ini.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Reviews", f"{len(filtered):,}")
            col2.metric("Avg Rating", f"{filtered['rating'].mean():.2f}")
            col3.metric("Avg Sentiment", f"{filtered['sentiment_score'].mean():.3f}")
            col4.metric("Unique Products", f"{filtered['product_name'].nunique():,}")

            st.subheader("Performa Produk")
            perf = (
                filtered.groupby("product_name", as_index=False)
                .agg(review_count=("review_id", "count"), avg_rating=("rating", "mean"), avg_sentiment=("sentiment_score", "mean"))
                .sort_values("review_count", ascending=False)
            )

            fig_perf = px.bar(
                perf.head(20),
                x="review_count",
                y="product_name",
                color="avg_sentiment",
                orientation="h",
                title="Top Products by Review Count",
                color_continuous_scale="Tealgrn",
            )
            st.plotly_chart(fig_perf, use_container_width=True)

            st.subheader("Sentiment vs Rating")
            heat = pd.crosstab(filtered["rating"], filtered["sentiment_label"], normalize="index").reset_index()
            heat_long = heat.melt(id_vars="rating", var_name="sentiment", value_name="ratio")
            fig_heat = px.density_heatmap(
                heat_long,
                x="sentiment",
                y="rating",
                z="ratio",
                color_continuous_scale="YlGnBu",
                title="Sentiment Proportion by Rating",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                sentiment_dist = filtered["sentiment_label"].value_counts().reset_index()
                sentiment_dist.columns = ["sentiment", "count"]
                fig_sent = px.pie(sentiment_dist, names="sentiment", values="count", title="Sentiment Share")
                st.plotly_chart(fig_sent, use_container_width=True)

            with c2:
                fig_box = px.box(filtered, x="rating", y="review_length", color="sentiment_label", title="Review Length by Rating")
                st.plotly_chart(fig_box, use_container_width=True)

            st.subheader("Wordcloud Sentimen")
            wc_scope = st.radio(
                "Cakupan wordcloud",
                ["Semua Produk", "Per Produk"],
                index=0,
                horizontal=True,
            )

            wc_df = filtered
            wc_title_suffix = "Semua Produk"
            if wc_scope == "Per Produk":
                selected_wc_product = st.selectbox("Pilih produk", sorted(filtered["product_name"].unique().tolist()))
                wc_df = filtered[filtered["product_name"] == selected_wc_product]
                wc_title_suffix = selected_wc_product

            wc_pos_text = sanitize_wordcloud_text(wc_df.loc[wc_df["sentiment_label"] == "positive", "comment_clean"])
            wc_neg_text = sanitize_wordcloud_text(wc_df.loc[wc_df["sentiment_label"] == "negative", "comment_clean"])

            w1, w2 = st.columns(2)
            with w1:
                render_wordcloud(wc_pos_text, f"Wordcloud Positif - {wc_title_suffix}")
            with w2:
                render_wordcloud(wc_neg_text, f"Wordcloud Negatif - {wc_title_suffix}")

            st.subheader("Ranking Produk")
            ranking_view = compute_ranking_view(filtered)
            st.dataframe(ranking_view[["product_name", "review_count", "avg_rating", "weighted_rating", "avg_sentiment", "csi", "segment"]], use_container_width=True)

            st.subheader("Review Explorer")
            cols = ["product_name", "username", "rating", "sentiment_label", "sentiment_score", "comment"]
            st.dataframe(filtered[cols].sort_values(["rating", "sentiment_score"], ascending=[True, True]), use_container_width=True)

        st.subheader("Insight Otomatis")
        if INSIGHTS_PATH.exists():
            st.markdown(INSIGHTS_PATH.read_text(encoding="utf-8"))
        else:
            st.info("Insight report belum dibuat.")

    with tab_test:
        st.subheader("Uji Sentimen dengan Model Aktif")
        if MODEL_INFO_PATH.exists():
            st.caption(MODEL_INFO_PATH.read_text(encoding="utf-8").strip())
        else:
            st.caption(f"model_id={MODEL_ID} | model_dir={MODEL_DIR}")

        input_text = st.text_area(
            "Masukkan teks review",
            height=140,
            placeholder="Contoh: Kopinya enak banget, aromanya kuat dan pengiriman cepat.",
        )
        run_btn = st.button("Prediksi Sentimen", type="primary")

        if run_btn:
            cleaned = normalize_text(input_text)
            if not cleaned:
                st.warning("Teks tidak boleh kosong.")
            else:
                with st.spinner("Memuat model dan melakukan prediksi..."):
                    try:
                        model = load_sentiment_model()
                        scores, labels = model.predict([cleaned])
                    except Exception as exc:
                        st.error(f"Gagal melakukan prediksi: {exc}")
                    else:
                        sentiment = labels[0]
                        score = float(scores[0])
                        st.success(f"Prediksi: {sentiment.upper()}")
                        st.metric("Sentiment Score (-1 s/d 1)", f"{score:.3f}")
                        st.progress(int((score + 1) * 50))
                        st.caption("Skor mendekati 1 artinya lebih positif, mendekati -1 artinya lebih negatif.")


if __name__ == "__main__":
    main()
