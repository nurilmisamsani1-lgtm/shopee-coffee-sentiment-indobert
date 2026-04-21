import os
import re
import glob
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoModelForSequenceClassification, AutoTokenizer


try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent
LIVE_DIR = ROOT_DIR / "live_product"
FIG_DIR = PROJECT_DIR / "figures"
MODEL_ID = "mdhugol/indonesia-bert-sentiment-classification"
MODEL_DIR = PROJECT_DIR / "models" / "indonesian_sentiment_mdhugol"

@dataclass
class IndonesianSentimentModel:
    model_id: str
    model_dir: Path
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification

    @staticmethod
    def _normalize_label(label: str) -> str:
        raw = str(label).strip().lower()
        if raw in {"label_0", "0"}:
            return "positive"
        if raw in {"label_1", "1"}:
            return "neutral"
        if raw in {"label_2", "2"}:
            return "negative"

        s = re.sub(r"[^a-z]", "", raw)
        if s in {"positive", "positif", "pos"}:
            return "positive"
        if s in {"negative", "negatif", "neg"}:
            return "negative"
        if s in {"neutral", "netral", "neu"}:
            return "neutral"
        return s

    @classmethod
    def load(cls, model_id: str = MODEL_ID, model_dir: Path = MODEL_DIR) -> "IndonesianSentimentModel":
        model_dir = Path(model_dir)
        local_ready = model_dir.exists() and any(model_dir.iterdir())
        source = str(model_dir) if local_ready else model_id

        tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_ready)
        model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=local_ready)

        if not local_ready:
            model_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

        return cls(
            model_id=model_id,
            model_dir=model_dir,
            tokenizer=tokenizer,
            model=model,
        )

    def predict(self, texts: list[str]) -> tuple[list[float], list[str]]:
        if not texts:
            return [], []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", {}) or {}
        class_labels = [self._normalize_label(id2label.get(i, str(i))) for i in range(self.model.config.num_labels)]

        pos_idx = next((i for i, c in enumerate(class_labels) if c == "positive"), None)
        neg_idx = next((i for i, c in enumerate(class_labels) if c == "negative"), None)
        if pos_idx is None or neg_idx is None:
            raise ValueError(f"Model labels are not compatible for sentiment scoring: {class_labels}")

        labels: list[str] = []
        scores = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for row in probs:
                pred_idx = int(np.argmax(row))
                labels.append(class_labels[pred_idx])
                pos = float(row[pos_idx])
                neg = float(row[neg_idx])
                scores.append(float(np.clip(pos - neg, -1.0, 1.0)))

        return scores, labels


def extract_product_name(path: Path) -> str:
    stem = path.stem
    prefix = "shopee_product_reviews_"
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"(?i)\bnan\b", " ", text)
    # Normalize elongated characters often found in chat/review language.
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_merge_csv() -> pd.DataFrame:
    files = sorted(glob.glob(str(LIVE_DIR / "shopee_product_reviews_*.csv")))
    if not files:
        raise FileNotFoundError(f"No matching CSV files found in {LIVE_DIR}")

    frames = []
    for fp in files:
        path = Path(fp)
        df = pd.read_csv(path)
        df["product_name"] = extract_product_name(path)
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    return master


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected_cols = [
        "review_id", "username", "rating", "date_variant", "likes",
        "image_count", "video_count", "comment", "product_name",
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    df["comment"] = df["comment"].fillna("").astype(str)
    df["username"] = df["username"].fillna("unknown").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating", "product_name"])
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    df = df.drop_duplicates(subset=["review_id", "product_name"], keep="first")

    # Remove known scraping noise and normalize text.
    df["comment_clean"] = df["comment"].str.replace("Membantu?", " ", regex=False)
    df["comment_clean"] = df["comment_clean"].apply(normalize_text)

    df["review_length"] = df["comment_clean"].str.split().str.len().fillna(0).astype(int)
    return df


def compute_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("product_name", as_index=False)
        .agg(
            review_count=("review_id", "count"),
            avg_rating=("rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_ratio=("sentiment_label", lambda s: (s == "positive").mean()),
            negative_ratio=("sentiment_label", lambda s: (s == "negative").mean()),
            avg_review_length=("review_length", "mean"),
        )
    )

    C = agg["avg_rating"].mean()
    m = max(1.0, agg["review_count"].quantile(0.6))
    v = agg["review_count"]
    R = agg["avg_rating"]
    agg["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

    # Customer Satisfaction Index in 0-100 scale.
    rating_norm = (agg["avg_rating"] - 1) / 4
    sent_norm = (agg["avg_sentiment"] + 1) / 2
    agg["csi"] = (0.6 * rating_norm + 0.4 * sent_norm) * 100

    global_rating = agg["avg_rating"].mean()
    global_sent = agg["avg_sentiment"].mean()
    agg["segment"] = "steady"
    agg.loc[(agg["avg_rating"] < global_rating) & (agg["avg_sentiment"] > global_sent), "segment"] = "undervalued"
    agg.loc[(agg["avg_rating"] > global_rating) & (agg["avg_sentiment"] < global_sent), "segment"] = "overrated"

    high_potential_mask = (
        (agg["review_count"] <= agg["review_count"].median())
        & (agg["avg_sentiment"] >= agg["avg_sentiment"].quantile(0.75))
        & (agg["avg_rating"] >= 4.0)
    )
    agg.loc[high_potential_mask, "segment"] = "high_potential"

    return agg.sort_values(["weighted_rating", "avg_sentiment"], ascending=False)


def save_visuals(df: pd.DataFrame, product_metrics: pd.DataFrame) -> None:
    try:
        import seaborn as sns
    except Exception:
        print("Warning: seaborn/scipy tidak tersedia, visual PNG dilewati.")
        sns = None

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")

    if sns is not None:
        # Reviews per product
        plt.figure(figsize=(14, 6))
        top_counts = product_metrics.sort_values("review_count", ascending=False).head(20)
        sns.barplot(data=top_counts, x="review_count", y="product_name", palette="viridis")
        plt.title("Top 20 Products by Review Count")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "bar_reviews_per_product.png", dpi=160)
        plt.close()

        # Rating distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(df["rating"], bins=5, kde=True, color="#2a9d8f")
        plt.title("Rating Distribution")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "distribution_rating.png", dpi=160)
        plt.close()

        # Sentiment vs rating heatmap
        heat = pd.crosstab(df["rating"], df["sentiment_label"], normalize="index")
        plt.figure(figsize=(8, 5))
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("Sentiment Proportion by Rating")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "heatmap_sentiment_rating.png", dpi=160)
        plt.close()

        # Boxplot of review length by rating
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="rating", y="review_length", palette="Set2")
        plt.title("Review Length by Rating")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "boxplot_review_length_rating.png", dpi=160)
        plt.close()

    # Trend chart by month
    monthly = df.copy()
    monthly["date"] = pd.to_datetime(monthly["date_variant"].str.extract(r"^(\d{4}-\d{2}-\d{2})", expand=False), errors="coerce")
    monthly = monthly.dropna(subset=["date"]) 
    if not monthly.empty:
        trend = monthly.groupby(monthly["date"].dt.to_period("M")).size().reset_index(name="reviews")
        trend["date"] = trend["date"].astype(str)
        fig = px.line(trend, x="date", y="reviews", title="Review Trend per Month")
        fig.write_html(str(FIG_DIR / "trend_reviews.html"), include_plotlyjs="cdn")

    # Product ranking interactive
    rank_plot = product_metrics.sort_values("weighted_rating", ascending=False).head(20)
    fig_rank = px.bar(
        rank_plot,
        x="weighted_rating",
        y="product_name",
        color="avg_sentiment",
        orientation="h",
        title="Top Products by Weighted Rating",
        color_continuous_scale="Tealgrn",
    )
    fig_rank.write_html(str(FIG_DIR / "ranking_weighted_rating.html"), include_plotlyjs="cdn")



def generate_wordclouds(df: pd.DataFrame) -> None:
    if not WORDCLOUD_AVAILABLE:
        return

    wc = WordCloud(width=1200, height=700, background_color="white", collocations=False)

    all_text = " ".join(df["comment_clean"].astype(str))
    if all_text.strip():
        wc.generate(all_text).to_file(str(FIG_DIR / "wordcloud_all.png"))

    pos_text = " ".join(df.loc[df["sentiment_label"] == "positive", "comment_clean"].astype(str))
    if pos_text.strip():
        wc.generate(pos_text).to_file(str(FIG_DIR / "wordcloud_positive.png"))

    neg_text = " ".join(df.loc[df["sentiment_label"] == "negative", "comment_clean"].astype(str))
    if neg_text.strip():
        wc.generate(neg_text).to_file(str(FIG_DIR / "wordcloud_negative.png"))

    for product, pdf in df.groupby("product_name"):
        ptxt = " ".join(pdf["comment_clean"].astype(str))
        if not ptxt.strip():
            continue
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", product)[:80]
        wc.generate(ptxt).to_file(str(FIG_DIR / f"wordcloud_{safe_name}.png"))


def get_top_ngrams(texts, n=2, top_k=20):
    vec = CountVectorizer(ngram_range=(n, n), min_df=2)
    X = vec.fit_transform(texts)
    sums = np.array(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    idx = np.argsort(sums)[::-1][:top_k]
    return list(zip(vocab[idx], sums[idx]))


def topic_modeling(df: pd.DataFrame, n_topics: int = 5):
    corpus = df["comment_clean"].fillna("")
    vec = CountVectorizer(max_df=0.95, min_df=3, stop_words="english")
    X = vec.fit_transform(corpus)
    if X.shape[0] < n_topics or X.shape[1] == 0:
        return []

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vec.get_feature_names_out()

    topics = []
    for idx, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[-10:][::-1]
        topics.append((idx + 1, ", ".join(words[top_idx])))
    return topics


def create_insights_report(df: pd.DataFrame, product_metrics: pd.DataFrame) -> None:
    top_review = product_metrics.sort_values("review_count", ascending=False).head(5)
    top_weighted = product_metrics.sort_values("weighted_rating", ascending=False).head(5)
    top_sent = product_metrics.sort_values("avg_sentiment", ascending=False).head(5)

    low_rating = df[df["rating"] <= 2]
    pain_tokens = Counter(" ".join(low_rating["comment_clean"]).split())
    top_pain = [w for w, c in pain_tokens.most_common(20) if len(w) > 2][:10]

    bi = get_top_ngrams(df["comment_clean"], n=2, top_k=20)
    tri = get_top_ngrams(df["comment_clean"], n=3, top_k=15)
    topics = topic_modeling(df, n_topics=5)

    corr_rating_sent = df[["rating", "sentiment_score"]].corr().iloc[0, 1]
    corr_len_rating = df[["review_length", "rating"]].corr().iloc[0, 1]

    best_product = product_metrics.sort_values("weighted_rating", ascending=False).iloc[0]
    worst_product = product_metrics.sort_values("weighted_rating", ascending=True).iloc[0]

    undervalued = product_metrics[product_metrics["segment"] == "undervalued"]["product_name"].head(5).tolist()
    overrated = product_metrics[product_metrics["segment"] == "overrated"]["product_name"].head(5).tolist()
    high_potential = product_metrics[product_metrics["segment"] == "high_potential"]["product_name"].head(5).tolist()

    lines = []
    lines.append("# Insights Report - Shopee Product Reviews\n")
    lines.append("## Executive Summary")
    lines.append(f"- Total review analyzed: **{len(df):,}**")
    lines.append(f"- Total product analyzed: **{df['product_name'].nunique():,}**")
    lines.append(f"- Correlation rating vs sentiment: **{corr_rating_sent:.3f}**")
    lines.append(f"- Correlation review length vs rating: **{corr_len_rating:.3f}**\n")

    lines.append("## Produk Terlaris (Proxy Jumlah Review)")
    for _, row in top_review.iterrows():
        lines.append(f"- {row['product_name']}: {int(row['review_count'])} reviews, avg rating {row['avg_rating']:.2f}")
    lines.append("")

    lines.append("## Produk Prioritas (Weighted Rating + Sentiment)")
    for _, row in top_weighted.iterrows():
        lines.append(f"- {row['product_name']}: weighted {row['weighted_rating']:.3f}, sentiment {row['avg_sentiment']:.3f}")
    lines.append("")

    lines.append("## Pain Points Customer")
    lines.append("- Keyword dominan pada rating rendah: " + ", ".join(top_pain if top_pain else ["tidak cukup data"]))
    lines.append("- Bigram populer: " + ", ".join([f"{w} ({int(c)})" for w, c in bi[:10]]))
    lines.append("- Trigram populer: " + ", ".join([f"{w} ({int(c)})" for w, c in tri[:8]]))
    lines.append("")

    lines.append("## Topic Modeling (LDA)")
    if topics:
        for tid, terms in topics:
            lines.append(f"- Topic {tid}: {terms}")
    else:
        lines.append("- Topic tidak terbentuk karena data teks terlalu pendek/sparse.")
    lines.append("")

    lines.append("## Segmentasi Produk")
    lines.append("- Undervalued: " + (", ".join(undervalued) if undervalued else "-"))
    lines.append("- Overrated: " + (", ".join(overrated) if overrated else "-"))
    lines.append("- High Potential: " + (", ".join(high_potential) if high_potential else "-"))
    lines.append("")

    lines.append("## Rekomendasi Strategi")
    lines.append(f"- Prioritaskan stok/promosi untuk produk top performer: **{best_product['product_name']}**.")
    lines.append(f"- Audit kualitas produk dengan skor terendah: **{worst_product['product_name']}**.")
    lines.append("- Gunakan pain-point keywords untuk perbaikan copywriting, QA produksi, dan FAQ customer support.")
    lines.append("- Monitor produk high potential dengan kampanye boost terarah (ads + bundling) untuk mendorong review volume.")
    lines.append("- Integrasikan score sentimen ke KPI mingguan agar sinyal kualitas terdeteksi sebelum rating turun signifikan.")

    (PROJECT_DIR / "insights_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> None:
    df = load_and_merge_csv()
    df = preprocess(df)

    model = IndonesianSentimentModel.load()
    scores, labels = model.predict(df["comment_clean"].fillna("").astype(str).tolist())
    df["sentiment_score"] = scores
    df["sentiment_label"] = labels

    product_metrics = compute_product_metrics(df)
    create_insights_report(df, product_metrics)
    save_visuals(df, product_metrics)
    generate_wordclouds(df)

    df.to_csv(PROJECT_DIR / "data_cleaned.csv", index=False)
    (PROJECT_DIR / "sentiment_model_source.txt").write_text(
        f"model_id={model.model_id}\nmodel_dir={model.model_dir}\n",
        encoding="utf-8",
    )

    product_metrics.to_csv(PROJECT_DIR / "product_metrics.csv", index=False)

    print("Pipeline finished successfully.")
    print(f"Rows: {len(df)}, Products: {df['product_name'].nunique()}")


if __name__ == "__main__":
    run_pipeline()
