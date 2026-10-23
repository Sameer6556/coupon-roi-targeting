"""A small fake version of The Complete Journey with the same 8 tables and schemas.

Used by the tests and by CI (`couponroi all --synthetic`) so the whole pipeline can run
end to end in about a minute without downloading anything. It is built to exercise the
code paths, not to look like the real data: heavier shoppers are more likely to get
Type A booklets (so naive comparisons are biased), the booklets have a small true effect
(TRUE_EFFECT dollars per week), and some households go quiet for a while.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_EFFECT = 3.0  # $/week while a Type A booklet is in hand
WEEK2_START = pd.Timestamp("2017-01-02")
DEPARTMENTS = [
    "GROCERY",
    "GROCERY",
    "GROCERY",
    "PRODUCE",
    "MEAT",
    "DRUG GM",
    "DELI",
    "PASTRY",
    "MEAT-PCKGD",
    "MISCELLANEOUS",
]
TYPE_A_WAVES = [
    ("2017-02-08", "2017-03-26"),
    ("2017-05-08", "2017-06-25"),
    ("2017-08-08", "2017-09-24"),
    ("2017-10-30", "2017-12-24"),
]


def _week_dates(week: int) -> pd.DatetimeIndex:
    if week == 1:
        return pd.DatetimeIndex([pd.Timestamp("2017-01-01")])
    start = WEEK2_START + pd.Timedelta(days=7 * (week - 2))
    days = pd.date_range(start, periods=7, freq="D")
    if week == 53:  # closed on Christmas Day, like the real data
        days = days[days != pd.Timestamp("2017-12-25")]
    return days


def _week_of(date: pd.Timestamp) -> int:
    return 2 + int(np.floor((date - WEEK2_START).days / 7))


def make_tables(n_households: int = 600, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    hh = np.arange(1, n_households + 1)

    # ---- products
    n_prod = 400
    products = pd.DataFrame(
        {
            "product_id": np.arange(100000, 100000 + n_prod),
            "manufacturer_id": rng.integers(1, 60, n_prod),
            "department": rng.choice(DEPARTMENTS, n_prod),
            "brand": rng.choice(["National", "Private"], n_prod, p=[0.8, 0.2]),
            "product_category": rng.choice([f"CAT {i}" for i in range(40)], n_prod),
            "product_type": rng.choice([f"TYPE {i}" for i in range(120)], n_prod),
            "package_size": rng.choice(["12 OZ", "1 LB", "16 OZ", None], n_prod),
        }
    )
    products.loc[products.index[:6], "department"] = "FUEL"
    products.loc[rng.choice(n_prod, 12, replace=False), ["product_category", "product_type"]] = None
    price = dict(zip(products.product_id, rng.lognormal(1.0, 0.6, n_prod).round(2), strict=True))
    fuel_ids = products.loc[products.department == "FUEL", "product_id"].to_numpy()
    shop_ids = products.loc[products.department != "FUEL", "product_id"].to_numpy()

    # ---- households: spend level, shopping frequency, a quiet spell for some
    level = rng.lognormal(3.3, 0.7, n_households)  # typical weekly spend
    freq = rng.beta(2.5, 1.5, n_households)  # chance of shopping in a given week
    quiet_start = np.where(rng.random(n_households) < 0.35, rng.integers(10, 50, n_households), 999)
    quiet_len = rng.integers(6, 20, n_households)
    coupon_user = rng.random(n_households) < 0.3

    # ---- campaigns: Type A waves target heavier shoppers; a few small Type B/C
    desc_rows, camp_rows = [], []
    cid = 1
    logit_base = -0.2 + 1.3 * (np.log(level) - np.log(level).mean())
    type_a_windows = []
    for s, e in TYPE_A_WAVES:
        p = 1 / (1 + np.exp(-logit_base))
        chosen = hh[rng.random(n_households) < p]
        desc_rows.append((cid, "Type A", pd.Timestamp(s), pd.Timestamp(e)))
        camp_rows += [(cid, h) for h in chosen]
        type_a_windows.append((cid, _week_of(pd.Timestamp(s)), _week_of(pd.Timestamp(e)), set(chosen)))
        cid += 1
    for s in ["2017-03-01", "2017-06-01", "2017-07-15", "2017-09-10", "2017-11-15"]:
        start = pd.Timestamp(s)
        ctype = "Type B" if rng.random() < 0.7 else "Type C"
        desc_rows.append((cid, ctype, start, start + pd.Timedelta(days=35)))
        camp_rows += [(cid, h) for h in hh[rng.random(n_households) < 0.12]]
        cid += 1
    campaign_descriptions = pd.DataFrame(
        desc_rows, columns=["campaign_id", "campaign_type", "start_date", "end_date"]
    )
    campaigns = pd.DataFrame(camp_rows, columns=["campaign_id", "household_id"])

    # ---- transactions
    rows = []
    basket = 30000000000
    stores = np.arange(300, 312)
    home_store = rng.choice(stores, n_households)
    for w in range(1, 54):
        days = _week_dates(w)
        for i, h in enumerate(hh):
            quiet = quiet_start[i] <= w < quiet_start[i] + quiet_len[i]
            p_shop = freq[i] * (0.1 if quiet else 1.0) * (0.3 if w == 1 else 1.0)
            if rng.random() > p_shop:
                continue
            target = level[i] * rng.lognormal(0, 0.35)
            for _cid, s, e, members in type_a_windows:
                if h in members and s <= w <= e:
                    target += TRUE_EFFECT * (2.0 if coupon_user[i] else 1.0) / max(p_shop, 0.2)
            n_trips = 1 + rng.poisson(0.4)
            for _ in range(n_trips):
                basket += 1
                day = days[rng.integers(len(days))]
                ts_local = day + pd.Timedelta(
                    hours=int(rng.integers(7, 22)), minutes=int(rng.integers(0, 60))
                )
                store = home_store[i] if rng.random() < 0.85 else rng.choice(stores)
                spend_left = target / n_trips
                while spend_left > 0.5:
                    pid = int(rng.choice(shop_ids))
                    qty = int(rng.integers(1, 3))
                    val = round(min(price[pid] * qty, spend_left + 1), 2)
                    disc = round(val * rng.choice([0, 0, 0.1, 0.2]), 2)
                    cdisc = round(val * 0.15, 2) if (coupon_user[i] and rng.random() < 0.05) else 0.0
                    rows.append((h, store, basket, pid, qty, val, disc, cdisc, 0.0, w, ts_local))
                    spend_left -= val
                if rng.random() < 0.12:
                    rows.append(
                        (
                            h,
                            store,
                            basket,
                            int(rng.choice(fuel_ids)),
                            int(rng.integers(5000, 15000)),
                            round(float(rng.uniform(20, 45)), 2),
                            0.0,
                            0.0,
                            0.0,
                            w,
                            ts_local,
                        )
                    )
    tx = pd.DataFrame(
        rows,
        columns=[
            "household_id",
            "store_id",
            "basket_id",
            "product_id",
            "quantity",
            "sales_value",
            "retail_disc",
            "coupon_disc",
            "coupon_match_disc",
            "week",
            "ts_local",
        ],
    )
    # stored like the real data: naive UTC timestamps, local time = America/New_York
    tx["transaction_timestamp"] = (
        tx.pop("ts_local")
        .dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous=False)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    tx["transaction_timestamp"] = tx["transaction_timestamp"].astype("datetime64[ns]")

    # ---- coupons and redemptions
    coupon_rows, red_rows = [], []
    upc = 50000000000
    for c in campaign_descriptions.itertuples():
        for _ in range(int(rng.integers(5, 15))):
            upc += 1
            for pid in rng.choice(shop_ids, int(rng.integers(3, 20)), replace=False):
                coupon_rows.append((upc, int(pid), c.campaign_id))
        members = campaigns.loc[campaigns.campaign_id == c.campaign_id, "household_id"].to_numpy()
        span = (c.end_date - c.start_date).days
        for h in members:
            if rng.random() < (0.25 if coupon_user[h - 1] else 0.05):
                for _ in range(int(rng.integers(1, 4))):
                    d = c.start_date + pd.Timedelta(days=int(rng.integers(0, span + 1)))
                    d = min(max(d, pd.Timestamp("2017-01-01")), pd.Timestamp("2017-12-31"))
                    red_rows.append((int(h), upc - int(rng.integers(0, 5)), c.campaign_id, d))
    coupons = pd.DataFrame(coupon_rows, columns=["coupon_upc", "product_id", "campaign_id"])
    redemptions = pd.DataFrame(
        red_rows, columns=["household_id", "coupon_upc", "campaign_id", "redemption_date"]
    )
    redemptions = redemptions[
        redemptions.apply(
            lambda r: (
                campaign_descriptions.set_index("campaign_id").loc[r.campaign_id, "start_date"]
                <= r.redemption_date
                <= campaign_descriptions.set_index("campaign_id").loc[r.campaign_id, "end_date"]
            ),
            axis=1,
        )
    ].reset_index(drop=True)

    # ---- promotions (display / mailer flags per product, store, week)
    n_promo = 6000
    promotions = pd.DataFrame(
        {
            "product_id": rng.choice(shop_ids, n_promo),
            "store_id": rng.choice(stores, n_promo),
            "display_location": rng.choice(["0", "1", "3", "5", "9", "A"], n_promo),
            "mailer_location": rng.choice(["0", "A", "D", "F"], n_promo),
            "week": rng.integers(1, 54, n_promo),
        }
    ).drop_duplicates(["product_id", "store_id", "week"])

    # ---- demographics for a non-random third (heavier shoppers are likelier to have them)
    has_demo = rng.random(n_households) < np.clip(0.15 + 0.4 * (level > np.median(level)), 0, 1)
    d_ids = hh[has_demo]
    demographics = pd.DataFrame(
        {
            "household_id": d_ids,
            "age": rng.choice(["19-24", "25-34", "35-44", "45-54", "55-64", "65+"], len(d_ids)),
            "income": rng.choice(
                ["Under 15K", "15-24K", "25-34K", "35-49K", "50-74K", "75-99K", "100-124K"], len(d_ids)
            ),
            "home_ownership": rng.choice(["Homeowner", "Renter", None], len(d_ids)),
            "marital_status": rng.choice(["Married", "Unmarried"], len(d_ids)),
            "household_size": rng.choice(["1", "2", "3", "4", "5+"], len(d_ids)),
            "household_comp": rng.choice(
                ["1 Adult No Kids", "2 Adults No Kids", "2 Adults Kids"], len(d_ids)
            ),
            "kids_count": rng.choice(["0", "1", "2", "3+"], len(d_ids)),
        }
    )
    for df in (products, demographics):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("string")
    for col in ("display_location", "mailer_location"):
        promotions[col] = promotions[col].astype("string")
    campaign_descriptions["campaign_type"] = campaign_descriptions["campaign_type"].astype("string")
    return {
        "transactions": tx,
        "products": products,
        "demographics": demographics,
        "campaigns": campaigns,
        "campaign_descriptions": campaign_descriptions,
        "coupons": coupons,
        "coupon_redemptions": redemptions,
        "promotions": promotions,
    }
