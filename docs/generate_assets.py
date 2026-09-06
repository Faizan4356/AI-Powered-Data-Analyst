"""Generates representative chart PNGs for the README from the real sample
data — same modules the app itself uses, exported via kaleido so the README
shows real computed output, not mockups.
"""
import sys

sys.path.insert(0, "..")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_dark"

from analysis import anomaly_detection, cohort_analysis, forecasting, root_cause_analysis, segmentation, significance_testing

OUT = "screenshots"
PALETTE = ["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#06B6D4", "#8B5CF6", "#EF4444"]

customers = pd.read_csv("../sample_data/customers.csv")
orders = pd.read_csv("../sample_data/orders.csv")
df = orders.merge(customers[["customer_id", "region"]].drop_duplicates("customer_id"), on="customer_id", how="left")
df["order_date"] = pd.to_datetime(df["order_date"])


def save(fig, name, height=420, width=760):
    fig.update_layout(height=height, width=width, margin=dict(l=40, r=20, t=50, b=40))
    fig.write_image(f"{OUT}/{name}.png", scale=2)
    print("saved", name)


# 1. Dataset comparison overview (bar)
overview = pd.DataFrame(
    {
        "table": ["customers", "orders"],
        "rows": [len(customers), len(orders)],
        "columns": [len(customers.columns), len(orders.columns)],
    }
).melt(id_vars="table", value_vars=["rows", "columns"], var_name="metric", value_name="value")
save(px.bar(overview, x="table", y="value", color="metric", barmode="group", text="value",
            title="Dataset Comparison: customers.csv vs orders.csv", color_discrete_sequence=PALETTE), "dashboard_comparison")

# 2. Forecast chart
fc = forecasting.run_forecast(df, "order_date", "revenue", periods=3, freq="ME")
hist = fc.history.assign(kind="history").rename(columns={"y": "value"})
fut = fc.forecast.assign(kind="forecast").rename(columns={"yhat": "value"})
combined = pd.concat([hist[["ds", "value", "kind"]], fut[["ds", "value", "kind"]]])
save(px.line(combined, x="ds", y="value", color="kind", title="Explainable Forecast: Monthly Revenue",
             color_discrete_sequence=PALETTE, markers=True), "forecast")

# 3. Anomaly detection chart (highlighting the injected $5000 outlier)
z = anomaly_detection.detect_zscore(df, "revenue", threshold=3.0)
scatter_df = df.copy()
scatter_df["is_anomaly"] = scatter_df.index.isin(z.flagged_rows.index)
save(px.scatter(scatter_df, x="order_date", y="revenue", color="is_anomaly",
                 title="Anomaly Detection: Revenue Outliers (Z-score)",
                 color_discrete_map={True: "#EF4444", False: "#6366F1"}), "anomalies")

# 4. Segmentation scatter (RFM + KMeans)
rfm = segmentation.compute_rfm(df, "customer_id", "order_date", "revenue")
seg = segmentation.kmeans_segment(rfm, n_clusters=4)
save(px.scatter(seg.labeled_table, x="recency", y="monetary", color="label", size="frequency",
                 title="RFM Customer Segmentation (K-Means)", color_discrete_sequence=PALETTE), "segmentation")

# 5. Cohort retention heatmap
cohort_result = cohort_analysis.build_retention_cohorts(df, "customer_id", "order_date", period="M")
save(px.imshow(cohort_result.retention_pct, text_auto=True, color_continuous_scale="Purples",
               title="Retention Cohort Analysis (%)"), "cohort_retention")

# 6. Root-cause driver chart (engineered East-region decline)
rc = root_cause_analysis.decompose_change(df, "order_date", "revenue", "region", "2024-01-01", "2024-04-30", "2024-05-01", "2024-06-30")
drivers_df = pd.DataFrame([vars(d) for d in rc.drivers])
save(px.bar(drivers_df, x="dimension_value", y="change", color="change", color_continuous_scale="RdYlGn",
            title="Root-Cause Analysis: Revenue Change by Region"), "root_cause")

# 7. A/B testing summary
ab = significance_testing.compare_means(df, "channel", "revenue", "email", "organic")
bar_df = pd.DataFrame({"channel": ["email", "organic"], "mean_revenue": [ab.group_a_mean, ab.group_b_mean]})
fig7 = px.bar(bar_df, x="channel", y="mean_revenue", color="channel", text="mean_revenue",
              title=f"A/B Testing: Email vs Organic (p={ab.p_value:.3f})", color_discrete_sequence=PALETTE)
save(fig7, "ab_testing")

print("\nAll assets generated in docs/screenshots/")
