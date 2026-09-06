"""Generates two related sample datasets for manually testing every tab of
the AI Data Analyst app: customers.csv and orders.csv, joinable on
customer_id, spanning 6 months with intentional data-quality issues,
an outlier, a regional revenue decline, and channel/region dimensions.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------------- customers.csv ----------------
n_customers = 40
regions = rng.choice(["East", "West", "North", "South"], size=n_customers, p=[0.35, 0.35, 0.15, 0.15])
signup_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 150, size=n_customers), unit="D")

customers = pd.DataFrame(
    {
        "customer_id": [f"CUST{i:03d}" for i in range(1, n_customers + 1)],
        "name": [f"Customer {i}" for i in range(1, n_customers + 1)],
        "email": [f"customer{i}@example.com" for i in range(1, n_customers + 1)],
        "region": regions,
        "signup_date": signup_dates.strftime("%Y-%m-%d"),
    }
)

# Introduce a few missing emails and one duplicate customer row (data-quality testing)
customers.loc[3, "email"] = None
customers.loc[7, "email"] = None
customers = pd.concat([customers, customers.iloc[[5]]], ignore_index=True)

customers.to_csv("customers.csv", index=False)

# ---------------- orders.csv ----------------
n_orders = 400
order_customer_ids = rng.choice(customers["customer_id"].iloc[:n_customers], size=n_orders)
order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 181, size=n_orders), unit="D")
products = rng.choice(["Widget", "Gadget", "Gizmo", "Doohickey"], size=n_orders)
channels = rng.choice(["email", "ads", "organic"], size=n_orders, p=[0.4, 0.35, 0.25])
quantity = rng.integers(1, 6, size=n_orders)
base_price = rng.normal(50, 12, size=n_orders).clip(10)
revenue = (base_price * quantity).round(2)

orders = pd.DataFrame(
    {
        "order_id": [f"ORD{i:04d}" for i in range(1, n_orders + 1)],
        "customer_id": order_customer_ids,
        "order_date": order_dates.strftime("%Y-%m-%d"),
        "product": products,
        "channel": channels,
        "quantity": quantity,
        "revenue": revenue,
    }
)

# Merge in region so a revenue decline can be engineered for root-cause analysis
orders = orders.merge(customers[["customer_id", "region"]].drop_duplicates("customer_id"), on="customer_id", how="left")

# Engineer a real revenue decline in the East region during May-June (root-cause test target)
decline_mask = (pd.to_datetime(orders["order_date"]) >= "2024-05-01") & (orders["region"] == "East")
orders.loc[decline_mask, "revenue"] = (orders.loc[decline_mask, "revenue"] * 0.5).round(2)

# Inject one clear anomaly (huge order) for anomaly-detection testing
orders.loc[orders.index[0], "revenue"] = 5000.0

# Introduce some missing values and duplicate rows for profiling/cleaning testing
orders.loc[10:14, "quantity"] = None
orders.loc[20:22, "product"] = None
orders = pd.concat([orders, orders.iloc[[0, 1]]], ignore_index=True)

orders = orders.drop(columns=["region"])  # keep as a joinable table, not pre-merged

orders.to_csv("orders.csv", index=False)

print(f"Wrote customers.csv ({len(customers)} rows) and orders.csv ({len(orders)} rows)")
