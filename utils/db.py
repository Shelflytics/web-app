from __future__ import annotations
import streamlit as st
import pandas as pd
from supabase import create_client, Client

# ---------- Client ----------
@st.cache_resource
def get_client() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["anon_key"]   # read-only with your SELECT policy
    return create_client(url, key)

# ---------- Data fetchers ----------
@st.cache_data(ttl=600)  # cache 10 minutes
def fetch_sales_superstore(limit: int | None = None) -> pd.DataFrame:
    """Pull all rows (or first N) from sales_superstore, paginated to avoid timeouts."""
    client = get_client()
    rows = []
    page_size = 1000
    page = 0
    while True:
        start = page * page_size
        end = start + page_size - 1
        data = (
            client.table("sales_superstore")
            .select("*")
            .order("row_id")                    # stable order
            .range(start, end)
            .execute()
            .data
        )
        if not data:
            break
        rows.extend(data)
        page += 1
        if limit and len(rows) >= limit:
            rows = rows[:limit]
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["ship_date"] = pd.to_datetime(df["ship_date"])
        df["sales"] = pd.to_numeric(df["sales"])
        # make postal_code a string for display/search (some codes start with 0 in other countries)
        if "postal_code" in df.columns:
            df["postal_code"] = df["postal_code"].astype("Int64").astype(str)
    return df

@st.cache_data(ttl=600)
def get_sku_catalog() -> pd.DataFrame:
    """Unique list of SKUs from sales table (read-only)."""
    df = fetch_sales_superstore()
    if df.empty:
        return df
    keep = ["product_id", "product_name", "category", "sub_category"]
    cat = (
        df[keep]
        .dropna(subset=["product_id"])
        .drop_duplicates()
        .rename(columns={
            "product_id": "sku_id",
            "product_name": "name",
            "sub_category": "sub_category",
        })
        .reset_index(drop=True)
    )
    return cat

@st.cache_data(ttl=600)
def get_outlets() -> pd.DataFrame:
    """Approximate outlets from postal/city/state with sales totals."""
    df = fetch_sales_superstore()
    if df.empty:
        return df
    base = (
        df[["postal_code", "city", "state", "region"]]
        .dropna(subset=["postal_code"])
        .drop_duplicates()
    )
    sales = df.groupby("postal_code", dropna=True)["sales"].sum().reset_index(name="total_sales")
    out = base.merge(sales, on="postal_code", how="left").sort_values("total_sales", ascending=False)
    return out
