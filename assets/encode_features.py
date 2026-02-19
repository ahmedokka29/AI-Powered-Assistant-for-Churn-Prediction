import pandas as pd


# ── 1. BINARY MAPPINGS ────────────────────────────────────────────────────────

BINARY_MAPS = {
    "gender":             {"male": 1,  "female": 0},
    "is_married":         {"yes": 1,   "no": 0},
    "dependents":         {"yes": 1,   "no": 0},
    "phone_service":      {"yes": 1,   "no": 0},
    "paperless_billing":  {"yes": 1,   "no": 0},
}


# ── 2. ORDINAL MAPPINGS ───────────────────────────────────────────────────────
# Order is meaningful:
#   - service features: no_service(0) < no(1) < yes(2)
#   - contract:         month-to-month(0) < one_year(1) < two_year(2)

SERVICE_ORDER = {"no_internet_service": 0,
                 "no_phone_service": 0, "no": 1, "yes": 2}

ORDINAL_MAPS = {
    "dual":               SERVICE_ORDER,
    "online_security":    SERVICE_ORDER,
    "online_backup":      SERVICE_ORDER,
    "device_protection":  SERVICE_ORDER,
    "tech_support":       SERVICE_ORDER,
    "streaming_tv":       SERVICE_ORDER,
    "streaming_movies":   SERVICE_ORDER,
    "contract": {
        "month_to_month": 0,
        "one_year":        1,
        "two_year":        2,
    },
}


# ── 3. ONE-HOT FEATURES ───────────────────────────────────────────────────────
# Defined explicitly so the exact columns are always predictable
# regardless of what values appear in a given batch.

OHE_CATEGORIES = {
    "internet_service": ["dsl", "fiber_optic", "no"],
    "payment_method":   [
        "bank_transfer_automatic",
        "credit_card_automatic",
        "electronic_check",
        "mailed_check",
    ],
}


# ── 4. TARGET COLUMN (binary) ─────────────────────────────────────────────────

TARGET_MAP = {"yes": 1, "no": 0}


# ── MAIN ENCODING FUNCTION ────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame, encode_target: bool = True) -> pd.DataFrame:
    """
    Encode all categorical features using pure dictionary/pandas logic.
    No sklearn objects are created or needed — safe for any downstream pipeline.

    Parameters
    ----------
    df             : raw DataFrame with original string columns
    encode_target  : whether to encode the 'churn' column (set False at inference)

    Returns
    -------
    Encoded DataFrame ready for modelling.
    """
    df = df.copy()

    # Normalize: lowercase + strip whitespace on all string columns
    str_cols = df.select_dtypes(include=["object", "str"]).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.lower().str.strip())

    # ── Binary ────────────────────────────────────────────────────────────────
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # ── Ordinal ───────────────────────────────────────────────────────────────
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # ── One-Hot ───────────────────────────────────────────────────────────────
    for col, categories in OHE_CATEGORIES.items():
        if col in df.columns:
            for cat in categories:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df.drop(columns=[col], inplace=True)

    # ── Target ───────────────────────────────────────────────────────────────
    if encode_target and "churn" in df.columns:
        df["churn"] = df["churn"].map(TARGET_MAP)

    return df


# ── OPTIONAL: DECODE BACK TO READABLE (useful for debugging) ─────────────────

def decode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse binary encoding for inspection — does not reverse OHE/ordinal."""
    df = df.copy()
    reverse = {col: {v: k for k, v in mapping.items()}
               for col, mapping in BINARY_MAPS.items()}
    for col, mapping in reverse.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


# ── QUICK SANITY CHECK ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.DataFrame([{
        "gender": "Male", "is_married": "Yes", "dependents": "No",
        "phone_service": "Yes", "dual": "No_phone_service",
        "internet_service": "Fiber_optic", "online_security": "No_internet_service",
        "online_backup": "No", "device_protection": "Yes",
        "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "No",
        "contract": "Month_to_month", "paperless_billing": "Yes",
        "payment_method": "Electronic_check", "churn": "No",
    }])

    encoded = encode_features(sample)

    print("Shape:", encoded.shape)
    print("\nColumns:", encoded.columns.tolist())
    print("\nValues:\n", encoded.T)
