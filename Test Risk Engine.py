from flask import Flask
from flask import request
from flask import jsonify

import math
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# =====================================================
# CONFIG
# =====================================================
OA_WEIGHTS = {
    "FraudLoss_Rate": 10,
    "ValueFlag_Rate": 5,
    "Void_Rate": 5,
    "Refund_Rate": 5,
    "BFF_Rate": 5,
    "MultipleTender_Rate": 5,
    "BeyondHours_Rate": 3,
    "ManagerMeal_Rate": 2,
}
SETTLEMENT_WEIGHTS = {
    "Cash_Variance_Ratio": 12,
    "Card_Variance_Ratio": 9,
    "Delivery_Variance_Ratio": 9,
}
FOOD_WEIGHTS = {
    "Unfavorable_Toplist_Gap": 15,
    "Unfavorable_Expected_Gap": 10,
}
FOOD_RAW_MAX = 25
FOOD_PILLAR_MAX = 30
# =====================================================
# HELPERS
# =====================================================
def percentile_rank(series):
    return series.rank(
        pct=True,
        method="average"
    ) * 100
def blended_percentile(df, value_col):
    enterprise_pct = percentile_rank(df[value_col])
    concept_pct = (
        df.groupby("Concept")[value_col]
        .transform(
            lambda x: x.rank(
                pct=True,
                method="average"
            ) * 100
        )
    )
    return (
        concept_pct * 0.70 +
        enterprise_pct * 0.30
    )
def risk_tier(score):
    if score >= 75:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Elevated"
    if score >= 20:
        return "Moderate"
    return "Low"
def confidence_label(coverage):
    if coverage >= 0.95:
        return "High Confidence"
    if coverage >= 0.60:
        return "Moderate Confidence"
    if coverage > 0:
        return "Limited Data"
    return "Insufficient Data"
# =====================================================
# STORE ELIGIBILITY (A3)
# =====================================================
def apply_exclusions(df):
    return df[
        (df["Concept"].str.upper() != "COMMISSARY")
        &
        (~df["Store"].str.upper().str.endswith("_CK"))
        &
        (df["Store"].str.upper() != "CENTRAL KITCHEN")
        &
        (df["Store"].str.upper() != "CENTRAL BAKERY")
    ].copy()
# =====================================================
# OA RATES
# =====================================================
def build_oa_rates(df):
    txn = (
        df["Transactions"]
        .replace(0, np.nan)
    )
    # Fraud Loss % / 1000 Txns
    df["FraudLoss_Rate"] = (
        df["FraudLoss_Pct"] * 1000
    )
    # Value Flag / 1000 Txns
    df["ValueFlag_Rate"] = (
        df["Infocode_Value_Flag"]
        / txn
        * 1000
    )
    # Void / 1000 Txns
    df["Void_Rate"] = (
        df["Void_Transactions"]
        / txn
        * 1000
    )
    # Refund / 1000 Txns
    df["Refund_Rate"] = (
        df["Refund_Transactions"]
        / txn
        * 1000
    )

    # BFF / 1000 Txns
    df["BFF_Rate"] = (
        df["Suspicious_BFF_Discount_Transactions"]
        / txn
        * 1000
    )

    # Multiple Tender / 1000 Txns
    df["MultipleTender_Rate"] = (
        (
            df["Bistro_GC_Other_Tender_Transactions"]
            + df["Irregular_MT_Cash_Amount"]
        )
        / txn
        * 1000
    )
    
    # Beyond Store Hours / 1000 Txns
    df["BeyondHours_Rate"] = (
        (
            df["NonSale_Beyond_Store_Hours"]
            + df["Sale_Beyond_Store_Hours"]
        )
        / txn
        * 1000
    )

    # Manager Meal / 1000 Txns
    df["ManagerMeal_Rate"] = (
        (
            df["Manager_Meal_Peak_Hours"]
            + df["Manager_Meal_Takeout"]
        )
        / txn
        * 1000
    )
    return df
# =====================================================
# OA SCORE
# =====================================================
def calculate_oa_score(df):
    score_cols = []
    for metric, max_points in OA_WEIGHTS.items():
        pct_col = f"{metric}_Pctile"
        pts_col = f"{metric}_Points"
        df[pct_col] = blended_percentile(
            df,
            metric
        )
        df[pts_col] = (
            df[pct_col]
            / 100
            * max_points
        )
        score_cols.append(pts_col)
    df["OA_Score"] = (
        df[score_cols]
        .sum(axis=1)
    )
    return df
# =====================================================
# SETTLEMENT
# =====================================================
def build_settlement_ratios(df):
    df["Cash_Variance_Ratio"] = np.where(
        df["Cash_POS"] > 0,
        df["Cash_Absolute_Daily_Variance"]
        /
        df["Cash_POS"],
        np.nan
    )
    df["Card_Variance_Ratio"] = np.where(
        df["Card_POS"] > 0,
        df["Card_Absolute_Daily_Variance"]
        /
        df["Card_POS"],
        np.nan
    )
    df["Delivery_Variance_Ratio"] = np.where(
        df["Delivery_POS"] > 0,
        df["Delivery_Absolute_Daily_Variance"]
        /
        df["Delivery_POS"],
        np.nan
    )
    return df
def calculate_settlement_score(df):
    score_cols = []
    for metric, max_points in SETTLEMENT_WEIGHTS.items():
        pct_col = f"{metric}_Pctile"
        pts_col = f"{metric}_Points"
        df[pct_col] = blended_percentile(
            df,
            metric
        )
        df[pts_col] = (
            df[pct_col]
            / 100
            * max_points
        )
        score_cols.append(pts_col)
    df["Settlement_Score"] = (
        df[score_cols]
        .sum(axis=1)
    )
    return df
# =====================================================
# FOOD COST
# =====================================================
def build_food_cost(df):
    df["Unfavorable_Toplist_Gap"] = np.maximum(
        0,
        df["Actual_Food_Cost_PctRevenue"]
        -
        df["Toplist_PctRevenue"]
    )
    df["Unfavorable_Expected_Gap"] = np.maximum(
        0,
        df["Actual_Food_Cost_PctRevenue"]
        -
        df["Expected_Toplist_PctRevenue"]
    )
    return df
def calculate_food_score(df):
    score_cols = []
    for metric, max_points in FOOD_WEIGHTS.items():
        pct_col = f"{metric}_Pctile"
        pts_col = f"{metric}_Points"
        df[pct_col] = blended_percentile(
            df,
            metric
        )
        df[pts_col] = (
            df[pct_col]
            / 100
            * max_points
        )
        score_cols.append(pts_col)
    df["Food_Raw_Points"] = (
        df[score_cols]
        .sum(axis=1)
    )
    # Normalize 25 -> 30
    df["FoodCost_Score"] = (
        df["Food_Raw_Points"]
        / FOOD_RAW_MAX
        * FOOD_PILLAR_MAX
    )
    return df
# =====================================================
# ENTERPRISE SCORE
# =====================================================
def calculate_enterprise(df):
    df["Earned_Points"] = (
        df["OA_Score"].fillna(0)
        +
        df["Settlement_Score"].fillna(0)
        +
        df["FoodCost_Score"].fillna(0)
    )
    df["Available_Points"] = 0
    df.loc[
        df["OA_Score"].notna(),
        "Available_Points"
    ] += 40
    df.loc[
        df["Settlement_Score"].notna(),
        "Available_Points"
    ] += 30
    df.loc[
        df["FoodCost_Score"].notna(),
        "Available_Points"
    ] += 30
    df["Coverage"] = (
        df["Available_Points"]
        / 100
    )
    # A11
    df["RiskScore"] = np.where(
        df["Available_Points"] > 0,
        (
            df["Earned_Points"]
            /
            df["Available_Points"]
        )
        * 100,
        np.nan
    )
    # A13
    df["RankingScore"] = (
        df["RiskScore"]
        *
        np.sqrt(df["Coverage"])
    )
    df["Confidence"] = (
        df["Coverage"]
        .apply(confidence_label)
    )
    df["RiskTier"] = (
        df["RiskScore"]
        .apply(risk_tier)
    )
    return df
# =====================================================
# PERCENTILES
# =====================================================
def calculate_final_percentiles(df):
    df["EnterprisePercentile"] = (
        percentile_rank(
            df["RiskScore"]
        )
    )
    df["ConceptPercentile"] = (
        df.groupby("Concept")["RiskScore"]
        .transform(
            lambda x:
                x.rank(
                    pct=True,
                    method="average"
                ) * 100
        )
    )
    return df
# =====================================================
# TOP DRIVERS
# =====================================================
DRIVER_NAMES = {
    "FraudLoss_Rate_Points": "Fraud Loss %",
    "ValueFlag_Rate_Points": "Value Flag",
    "Void_Rate_Points": "Void",
    "Refund_Rate_Points": "Refund",
    "BFF_Rate_Points": "BFF",
    "MultipleTender_Rate_Points": "Multiple Tender",
    "BeyondHours_Rate_Points": "Beyond Store Hours",
    "ManagerMeal_Rate_Points": "Manager's Meal",
    "Cash_Variance_Ratio_Points": "Cash Variance",
    "Card_Variance_Ratio_Points": "Card Variance",
    "Delivery_Variance_Ratio_Points": "Delivery Variance",
    "Unfavorable_Toplist_Gap_Points":
        "Food Cost vs Toplist",
    "Unfavorable_Expected_Gap_Points":
        "Food Cost vs Expected"
}
def top_drivers(row):
    scores = []
    for col, name in DRIVER_NAMES.items():
        if col in row:
            scores.append({
                "Indicator": name,
                "Points": round(row[col], 2)
            })
    return sorted(
        scores,
        key=lambda x: x["Points"],
        reverse=True
    )[:5]
# =====================================================
# MAIN PIPELINE
# =====================================================
def build_enterprise_risk_engine(df):
    df = apply_exclusions(df)
    df = build_oa_rates(df)
    df = calculate_oa_score(df)
    df = build_settlement_ratios(df)
    df = calculate_settlement_score(df)
    df = build_food_cost(df)
    df = calculate_food_score(df)
    df = calculate_enterprise(df)
    df = calculate_final_percentiles(df)
    df["TopDrivers"] = (
        df.apply(top_drivers, axis=1)
    )
    df = df.sort_values(
        "RankingScore",
        ascending=False
    ).reset_index(drop=True)
    df["Rank"] = df.index + 1
    return df
# =====================================================
# PAYLOAD LOADER
# =====================================================
def load_payload(payload):
    reporting_period = payload.get(
        "reporting_period",
        {}
    )
    report_year = reporting_period.get(
        "year"
    )
    report_month = reporting_period.get(
        "month_number"
    )
    mapping_df = pd.DataFrame(
        payload.get(
            "store_mapping",
            []
        )
    )
    oa_df = pd.DataFrame(
        payload.get(
            "operational_audit",
            []
        )
    )
    settlement_df = pd.DataFrame(
        payload.get(
            "settlement",
            []
        )
    )
    food_df = pd.DataFrame(
        payload.get(
            "food_cost",
            []
        )
    )
    return (
        report_year,
        report_month,
        mapping_df,
        oa_df,
        settlement_df,
        food_df
    )
# =====================================================
# COLUMN CLEANUP
# =====================================================
def clean_columns(df):
    df.columns = [
        str(col)
        .replace("[", "")
        .replace("]", "")
        .replace("STORE_MAPPING", "")
        .replace("VW_LSS_ENRICHED_W_AUDIT_FIELDS", "")
        .strip()
        for col in df.columns
    ]
    return df
# =====================================================
# DATA MERGE
# =====================================================
def merge_datasets(
    oa_df,
    settlement_df,
    food_df,
    mapping_df
):
    oa_df = clean_columns(oa_df)
    settlement_df = clean_columns(settlement_df)
    food_df = clean_columns(food_df)
    mapping_df = clean_columns(mapping_df)
    oa = oa_df.rename(columns={
        "STORE_ULTIMATE": "Store",
        "CONCEPT_ULTIMATE": "Concept",
        "VP/COO": "VP"
    })
    settlement = settlement_df.rename(columns={
        "Store360Name": "Store"
    })
    food = food_df.rename(columns={
        "Store": "Store",
        "Concept_": "Concept",
        "VP/COO_": "VP"
    })
    oa["Store"] = (
        oa["Store"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    settlement["Store"] = (
        settlement["Store"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    food["Store"] = (
        food["Store"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    merged = (
        oa
        .merge(
            settlement,
            on="Store",
            how="left",
            suffixes=("", "_SET")
        )
        .merge(
            food,
            on="Store",
            how="left",
            suffixes=("", "_FC")
        )
    )
    return merged

# =====================================================
# API RESPONSE
# =====================================================

def safe_float(value):
    ...

def build_store_response(row):
    ...
    return {
       ...
    }

# ADD THIS HERE
def build_api_response(
    results_df,
    report_year,
    report_month
):

    scored_stores = [
        build_store_response(row)
        for _, row
        in results_df.iterrows()
    ]

    return {
        "mode": "score_only",

        "reporting_period": {
            "year": report_year,
            "month_number": report_month
        },

        "store_count":
            len(scored_stores),

        "critical_count":
            sum(
                1
                for x in scored_stores
                if x["risk_tier"] == "Critical"
            ),

        "high_count":
            sum(
                1
                for x in scored_stores
                if x["risk_tier"] == "High"
            ),

        "elevated_count":
            sum(
                1
                for x in scored_stores
                if x["risk_tier"] == "Elevated"
            ),

        "moderate_count":
            sum(
                1
                for x in scored_stores
                if x["risk_tier"] == "Moderate"
            ),

        "low_count":
            sum(
                1
                for x in scored_stores
                if x["risk_tier"] == "Low"
            ),

        "scored_stores":
            scored_stores
    }

# =====================================================
# FLASK ENDPOINT
# =====================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "enterprise-risk-engine"
    })


@app.route("/", methods=["POST"])
def calculate_risk():

    try:

        payload = request.get_json(silent=True)

        if not payload:
            return jsonify({
                "status": "error",
                "message": "Request body is empty"
            }), 400

        mode = payload.get("mode", "score_only")

        if mode not in [
            "score_only",
            "generate_report"
        ]:
            return jsonify({
                "status": "error",
                "message": f"Unsupported mode: {mode}"
            }), 400

# =========================================
# 8.2 GENERATE REPORT
# =========================================

if mode == "generate_report":

    if "scored_stores" not in payload:
        return jsonify({
            "status": "error",
            "message": "Missing scored_stores"
        }), 400

    scored_stores = payload.get(
        "scored_stores",
        []
    )

    reporting_period = payload.get(
        "reporting_period",
        {}
    )

    return jsonify({

        "success": True,

        "mode": "generate_report",

        "report_success": True,

        "model": "gemini-3.6-flash",

        "reporting_period": {
            "year":
                reporting_period.get("year"),
            "month_number":
                reporting_period.get(
                    "month_number"
                )
        },

        "store_count":
            len(scored_stores),

        "report_html":
            "<h2>Report Generation Placeholder</h2>",

        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0
        }
    })


        # =========================================
        # 8.1 SCORE ONLY
        # =========================================
        required_keys = [
            "reporting_period",
            "store_mapping",
            "operational_audit",
            "settlement",
            "food_cost"
        ]

        missing = [
            key
            for key in required_keys
            if key not in payload
        ]

        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing payload sections: {missing}"
            }), 400

        (
            report_year,
            report_month,
            mapping_df,
            oa_df,
            settlement_df,
            food_df
        ) = load_payload(payload)

        print("OA Columns")
        print(oa_df.columns.tolist())

        print("Settlement Columns")
        print(settlement_df.columns.tolist())

        print("Food Cost Columns")
        print(food_df.columns.tolist())

        print("Mapping Columns")
        print(mapping_df.columns.tolist())

        merged_df = merge_datasets(
            oa_df,
            settlement_df,
            food_df,
            mapping_df
        )

        results_df = build_enterprise_risk_engine(
            merged_df
        )

        results_df = results_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        results_df = results_df.where(
            pd.notnull(results_df),
            None
        )

        response = build_api_response(
            results_df,
            report_year,
            report_month
        )

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                8080
            )
        )
    )
