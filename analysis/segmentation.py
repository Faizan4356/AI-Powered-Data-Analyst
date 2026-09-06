"""RFM scoring + K-Means customer segmentation with business-friendly labels."""
from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class SegmentationResult:
    rfm_table: pd.DataFrame
    cluster_profiles: pd.DataFrame
    labeled_table: pd.DataFrame
    n_clusters: int


def compute_rfm(df: pd.DataFrame, customer_col: str, date_col: str, amount_col: str, reference_date=None) -> pd.DataFrame:
    if len({customer_col, date_col, amount_col}) < 3:
        raise ValueError("Customer ID, date, and amount must be three distinct columns")

    working = df[[customer_col, date_col, amount_col]].dropna().copy()
    working[date_col] = pd.to_datetime(working[date_col])
    reference_date = reference_date or working[date_col].max() + pd.Timedelta(days=1)

    rfm = working.groupby(customer_col).agg(
        recency=(date_col, lambda x: (reference_date - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum"),
    ).reset_index()
    return rfm


def kmeans_segment(rfm: pd.DataFrame, n_clusters: int = 4, customer_col: str = None) -> SegmentationResult:
    features = rfm[["recency", "frequency", "monetary"]]
    scaled = StandardScaler().fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(scaled)

    labeled = rfm.copy()
    labeled["cluster"] = clusters

    profiles = labeled.groupby("cluster")[["recency", "frequency", "monetary"]].mean().reset_index()
    profiles["label"] = profiles.apply(_label_segment, axis=1, profiles=profiles)

    labeled = labeled.merge(profiles[["cluster", "label"]], on="cluster", how="left")

    return SegmentationResult(rfm_table=rfm, cluster_profiles=profiles, labeled_table=labeled, n_clusters=n_clusters)


def _label_segment(row, profiles: pd.DataFrame) -> str:
    r_med, f_med, m_med = profiles["recency"].median(), profiles["frequency"].median(), profiles["monetary"].median()
    if row["recency"] <= r_med and row["frequency"] >= f_med and row["monetary"] >= m_med:
        return "Champions"
    if row["recency"] > r_med and row["frequency"] >= f_med:
        return "At Risk"
    if row["recency"] <= r_med and row["frequency"] < f_med:
        return "New / Low Engagement"
    return "Needs Attention"
