import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ CREDIT CARD ------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #


def compute_first_agg_credit_card(df_cre: pd.DataFrame) -> pd.DataFrame:
    # We will now proceed to the first aggregation, this one will take place over SK_ID_PREV
    # We first compute the ratio

    df_cre["TOTAL_DRAWING_SUM"] = (
        df_cre["AMT_DRAWINGS_ATM_CURRENT"]
        + df_cre["AMT_DRAWINGS_CURRENT"]
        + df_cre["AMT_DRAWINGS_OTHER_CURRENT"]
        + df_cre["AMT_DRAWINGS_POS_CURRENT"]
    )
    df_cre["BALANCE_LIMIT_RATIO"] = df_cre["AMT_BALANCE"] / (
        df_cre["AMT_CREDIT_LIMIT_ACTUAL"] + 0.0001
    )
    df_cre["TOT_CNT_DRAWING"] = (
        df_cre["CNT_DRAWINGS_ATM_CURRENT"]
        + df_cre["CNT_DRAWINGS_CURRENT"]
        + df_cre["CNT_DRAWINGS_OTHER_CURRENT"]
        + df_cre["CNT_DRAWINGS_POS_CURRENT"]
        + df_cre["CNT_INSTALMENT_MATURE_CUM"]
    )
    df_cre["RATIO_DPD_DEF_DPD"] = df_cre["SK_DPD_DEF"] / (df_cre["SK_DPD"] + 0.0001)
    df_cre["AMT_INTEREST_RECEIVABLE"] = (
        df_cre["AMT_TOTAL_RECEIVABLE"] - df_cre["AMT_RECEIVABLE_PRINCIPAL"]
    )
    df_cre["PAYMENT_RECEIVABLE_RATIO"] = df_cre["AMT_TOTAL_RECEIVABLE"] / (
        df_cre["AMT_PAYMENT_TOTAL_CURRENT"] + 0.0001
    )

    # We then order the dataset using MONTHS_BALANCE
    df_cre.sort_values(by="MONTHS_BALANCE", ascending=False, inplace=True)

    # We compute LAST_STATUS_POS in the most efficient way (first() much slower)
    last_status_cre = (
        df_cre.groupby(["SK_ID_PREV", "SK_ID_CURR"])["NAME_CONTRACT_STATUS"]
        .nth(0)
        .rename("LAST_STATUS_CRE")
    )

    last_status_cre = (
        pd.concat(
            [
                df_cre.loc[last_status_cre.index][["SK_ID_PREV", "SK_ID_CURR"]],
                last_status_cre,
            ],
            axis=1,
        )
        .sort_values(by=["SK_ID_PREV", "SK_ID_CURR"])
        .set_index(["SK_ID_PREV", "SK_ID_CURR"])
    )

    # Now we can proceed with the temporal ones
    time_windows = [
        ("EVER", df_cre),
        ("LAST_YEAR", df_cre[df_cre["MONTHS_BALANCE"].between(-12, -1)]),
        ("LAST_3_MONTHS", df_cre[df_cre["MONTHS_BALANCE"] > -3]),
    ]
    diff_features = []
    all_window_columns = []
    for window_name, window_df in time_windows:
        agg_dict_diff = {
            "TOTAL_DRAWING_SUM": ["sum", "mean"],  # mean added
            "AMT_BALANCE": ["sum", "mean"],  # mean added
            "TOT_CNT_DRAWING": ["sum", "mean"],  # mean added
            "SK_DPD": ["max", "mean"],  # mean added
            "SK_DPD_DEF": ["max", "mean"],  # mean added
            "RATIO_DPD_DEF_DPD": ["max", "mean"],  # mean added
            "BALANCE_LIMIT_RATIO": ["max", "mean"],  # max added
            "AMT_PAYMENT_TOTAL_CURRENT": ["sum", "mean"],  # mean added
            "AMT_TOTAL_RECEIVABLE": ["sum", "mean"],  # mean added
            "PAYMENT_RECEIVABLE_RATIO": ["sum", "mean"],  # sum added
            "AMT_INTEREST_RECEIVABLE": ["sum", "mean"],  # sum added
        }
        window_features = window_df.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg(
            agg_dict_diff
        )
        window_features.columns = [
            f"{col[0]}_{window_name.upper()}_{col[1].upper()}"
            for col in window_features.columns.values
        ]
        diff_features.append(window_features)
        all_window_columns += list(window_features.columns.values)

    # We can combine all the features, reset the index, fill the missing values for the temporal features by 0 and drop the SK_ID_PREV
    first_agg_features = (
        last_status_cre.join(diff_features).reset_index().drop("SK_ID_PREV", axis=1)
    )
    first_agg_features[all_window_columns] = first_agg_features[
        all_window_columns
    ].fillna(0)

    return first_agg_features


def compute_second_agg_credit_card(first_agg_features: pd.DataFrame) -> pd.DataFrame:
    dum_status = pd.get_dummies(
        first_agg_features["LAST_STATUS_CRE"], dtype=np.int8, prefix="LAST_STATUS_CRE"
    )

    first_agg_features = pd.concat([dum_status, first_agg_features], axis=1)
    first_agg_features.drop("LAST_STATUS_CRE", axis=1, inplace=True)

    # Get the columns for different features
    status_columns = [
        col for col in first_agg_features.columns if col.startswith("LAST_STATUS")
    ]
    temporal_columns = [
        col
        for col in first_agg_features.columns
        if any(p in col for p in ["EVER", "LAST_YEAR", "LAST_3_MONTHS"])
    ]

    # Create the aggregation dictionaries
    agg_dict = {col: ["sum", "mean"] for col in status_columns}  # mean added
    agg_temp = {col: ["mean", "sum", "max"] for col in temporal_columns}  # sum,
    # max, added
    agg_dict.update(agg_temp)

    # Perform the second aggregation
    second_agg_features = first_agg_features.groupby("SK_ID_CURR").agg(agg_dict)
    second_agg_features.columns = [
        "CRE_BAL_" + "_".join(col).upper() for col in second_agg_features.columns.values
    ]

    return second_agg_features.reset_index()


def compute_features_credit_card(df_cre: pd.DataFrame) -> pd.DataFrame:
    df_cre_ = df_cre.copy()

    # Compute the first agg features (over SK_ID_PREV)
    first_agg = compute_first_agg_credit_card(df_cre_)

    # Compute the second agg features (over SK_ID_CURR)
    second_agg = compute_second_agg_credit_card(first_agg)

    # Reset the index to be SK_ID_CURR
    second_agg.set_index("SK_ID_CURR", inplace=True)

    return second_agg


# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------ PREVIOUS --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def cleaning_df_previous(df_previous: pd.DataFrame) -> pd.DataFrame:
    df_previous.loc[
        df_previous["DAYS_FIRST_DRAWING"] == 365243.0, "DAYS_FIRST_DRAWING"
    ] = np.nan
    df_previous.loc[
        df_previous["DAYS_FIRST_DUE"] == 365243.0, "DAYS_FIRST_DUE"
    ] = np.nan
    df_previous.loc[
        df_previous["DAYS_LAST_DUE_1ST_VERSION"] == 365243.0,
        "DAYS_LAST_DUE_1ST_VERSION",
    ] = np.nan
    df_previous.loc[df_previous["DAYS_LAST_DUE"] == 365243.0, "DAYS_LAST_DUE"] = np.nan
    df_previous.loc[
        df_previous["DAYS_TERMINATION"] == 365243.0, "DAYS_TERMINATION"
    ] = np.nan
    df_previous.loc[
        df_previous["SELLERPLACE_AREA"] > 3000000, "SELLERPLACE_AREA"
    ] = np.nan

    return df_previous


def compute_agg_previous_feature(df_pre: pd.DataFrame) -> pd.DataFrame:
    # We compute the necessary additional features
    df_pre["AMT_MISSING_CREDIT"] = df_pre["AMT_APPLICATION"] - df_pre["AMT_CREDIT"]
    df_pre["AMT_CREDIT_GOODS_DIFF"] = df_pre["AMT_GOODS_PRICE"] - df_pre["AMT_CREDIT"]
    df_pre["DIFF_CREDIT_DOWN_PAYMENT"] = (
        df_pre["AMT_CREDIT"] - df_pre["AMT_DOWN_PAYMENT"]
    )
    df_pre["AMT_INTEREST_LOAN"] = (
        df_pre["CNT_PAYMENT"] * df_pre["AMT_ANNUITY"] - df_pre["AMT_CREDIT"]
    )
    df_pre["LOANS_DURATION"] = df_pre["DAYS_TERMINATION"] - df_pre["DAYS_FIRST_DUE"]

    # Using dummies on categorical features
    dum_name_payment = pd.get_dummies(
        df_pre["NAME_PAYMENT_TYPE"], dtype=np.int8, prefix="PRE_NAME_PAYMENT_TYPE"
    )
    dum_name_payment.columns = [
        col.replace(" ", "_").upper() for col in dum_name_payment.columns.values
    ]

    dum_product_combination = pd.get_dummies(
        df_pre["PRODUCT_COMBINATION"], dtype=np.int8, prefix="PRE_PRODUCT_COMBINATION"
    )
    dum_product_combination.columns = [
        col.replace(" ", "_").replace(":", "").upper()
        for col in dum_product_combination.columns.values
    ]

    dum_name_portfolio = pd.get_dummies(
        df_pre["NAME_PORTFOLIO"], dtype=np.int8, prefix="PRE_NAME_PORTFOLIO"
    )
    dum_name_portfolio.columns = [
        col.replace(" ", "_").upper() for col in dum_name_portfolio.columns.values
    ]

    code_reject_reason = pd.get_dummies(
        df_pre["CODE_REJECT_REASON"], dtype=np.int8, prefix="PRE_CODE_REJECT_REASON"
    )
    code_reject_reason.columns = [
        col.replace(" ", "_").upper() for col in code_reject_reason.columns.values
    ]

    df_pre = pd.concat(
        [
            df_pre,
            dum_name_payment,
            dum_product_combination,
            dum_name_portfolio,
            code_reject_reason,
        ],
        axis=1,
    )
    cat_columns = [col for col in df_pre.columns if col.startswith("PRE_")]
    agg_dict_cat = {col: "sum" for col in cat_columns}

    agg_dict = {
        "AMT_CREDIT": ["sum", "mean", "max"],  # added
        "DAYS_FIRST_DUE": ["min", "mean"],  # added
        "AMT_ANNUITY": ["sum", "mean", "max"],
        "AMT_MISSING_CREDIT": ["sum", "mean"],
        "AMT_CREDIT_GOODS_DIFF": ["sum", "mean", "max"],  # max added
        "DIFF_CREDIT_DOWN_PAYMENT": ["sum", "mean", "max"],  # max added
        "AMT_INTEREST_LOAN": ["sum", "mean", "max"],  # max added
        "DAYS_DECISION": ["mean", "min"],
        "DAYS_TERMINATION": ["mean", "max"],
        "LOANS_DURATION": ["mean", "max", "min"],  # min added
    }

    agg_dict.update(agg_dict_cat)

    previous_features = df_pre.groupby("SK_ID_CURR").agg(agg_dict)
    previous_features.columns = [
        "PREVIOUS_" + "_".join(col).upper() for col in previous_features.columns.values
    ]

    return previous_features


def compute_features_previous(df_pre: pd.DataFrame) -> pd.DataFrame:
    df_pre_ = df_pre.copy()

    # We first clean the dataset to remove anomalies
    df_pre_ = cleaning_df_previous(df_pre_)

    # Compute the second agg features (over SK_ID_CURR)
    pre_features = compute_agg_previous_feature(df_pre_)

    return pre_features


# ------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- BUREAU --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def clean_df_bureau(df_bur: pd.DataFrame) -> pd.DataFrame:
    df_bur.loc[
        np.abs(df_bur["DAYS_CREDIT_ENDDATE"]) > 15000, "DAYS_CREDIT_ENDDATE"
    ] = np.nan
    df_bur.loc[
        np.abs(df_bur["DAYS_ENDDATE_FACT"]) > 15000, "DAYS_ENDDATE_FACT"
    ] = np.nan
    df_bur.loc[
        np.abs(df_bur["DAYS_CREDIT_UPDATE"]) > 15000, "DAYS_CREDIT_UPDATE"
    ] = np.nan
    df_bur.loc[
        np.abs(df_bur["AMT_CREDIT_MAX_OVERDUE"]) > 1e8, "AMT_CREDIT_MAX_OVERDUE"
    ] = np.nan
    df_bur.loc[np.abs(df_bur["AMT_CREDIT_SUM"]) > 1e8, "AMT_CREDIT_SUM"] = np.nan
    df_bur.loc[
        np.abs(df_bur["AMT_CREDIT_SUM_DEBT"]) > 5e7, "AMT_CREDIT_SUM_DEBT"
    ] = np.nan

    return df_bur


def compute_agg_bureau_features(df_bur: pd.DataFrame) -> pd.DataFrame:
    # We compute the necessary additional features
    df_bur["CREDIT_DURATION"] = df_bur["DAYS_CREDIT_ENDDATE"] - df_bur["DAYS_CREDIT"]
    df_bur["EFFECTIVE_CREDIT_DURATION"] = (
        df_bur["DAYS_ENDDATE_FACT"] - df_bur["DAYS_CREDIT"]
    )
    df_bur["RATIO_ANNUITY_AMT_CREDIT"] = df_bur["AMT_ANNUITY"] / (
        df_bur["AMT_CREDIT_SUM"] + 0.0001
    )
    df_bur["RATIO_CURRENT_DEBT_CREDIT"] = df_bur["AMT_CREDIT_SUM_DEBT"] / (
        df_bur["AMT_CREDIT_SUM"] + 0.0001
    )
    df_bur["RATIO_CNT_PROLONGED_DURATION"] = df_bur["CNT_CREDIT_PROLONG"] / (
        df_bur["CREDIT_DURATION"] + 0.0001
    )
    df_bur["RATIO_AMT_CREDIT_LIMIT"] = df_bur["AMT_CREDIT_SUM"] / (
        df_bur["AMT_CREDIT_SUM_LIMIT"] + 0.0001
    )
    df_bur["RATION_SUM_OVERDUE_CREDIT"] = df_bur["AMT_CREDIT_SUM_OVERDUE"] / (
        df_bur["AMT_CREDIT_SUM"] + 0.0001
    )

    credit_type_cats = [
        "Consumer credit",
        "Credit card",
        "Car loan",
        "Microloan",
        "Mortgage",
    ]
    df_bur["CREDIT_TYPE_MOD"] = df_bur["CREDIT_TYPE"].apply(
        lambda x: x if x in credit_type_cats else "OTHER"
    )
    credit_type_dummies = pd.get_dummies(
        df_bur["CREDIT_TYPE_MOD"], dtype=np.int8, prefix="CREDIT_TYPE"
    )
    credit_type_dummies.columns = [
        col.replace(" ", "_").upper() for col in credit_type_dummies.columns.values
    ]
    agg_credit_type = {
        col: ["sum", "mean", "max"] for col in credit_type_dummies.columns
    }

    df_bur.drop("CREDIT_TYPE", axis=1, inplace=True)
    df_bur = pd.concat([df_bur, credit_type_dummies], axis=1)

    all_features = []
    cats_to_agg = ["Active", "Closed"]
    for cat in cats_to_agg:
        df_filtered = df_bur[df_bur["CREDIT_ACTIVE"] == cat]

        agg_dict = {
            "DAYS_CREDIT": ["min", "max", "mean"],  # mean added
            "AMT_CREDIT_SUM": ["sum", "max", "std"],  # std added
            "AMT_CREDIT_SUM_DEBT": ["sum", "max"],
            "CREDIT_DURATION": ["mean", "min", "max"],
            "EFFECTIVE_CREDIT_DURATION": ["mean", "min", "max"],
            "RATIO_ANNUITY_AMT_CREDIT": ["mean", "max"],  # max added
            "RATIO_CURRENT_DEBT_CREDIT": ["mean", "max"],  # max added
            "RATIO_CNT_PROLONGED_DURATION": ["mean", "max"],  # max added
            "RATIO_AMT_CREDIT_LIMIT": ["mean", "max"],  # max added
            "RATION_SUM_OVERDUE_CREDIT": ["mean", "max"],  # max added
        }
        agg_dict.update(agg_credit_type)

        tmp_bureau_features = df_bur.groupby("SK_ID_CURR").agg(agg_dict)
        tmp_bureau_features.columns = [
            "BUREAU_" + cat.upper() + "_" + "_".join(col).upper()
            for col in tmp_bureau_features.columns.values
        ]
        all_features.append(tmp_bureau_features)

    bureau_features = pd.concat(all_features, axis=1)
    return bureau_features.fillna(0)


def compute_features_bureau(df_bur: pd.DataFrame) -> pd.DataFrame:
    df_bur_ = df_bur.copy()

    # We first clean the dataset to remove anomalies
    df_bur_ = clean_df_bureau(df_bur_)

    # Compute the second agg features (over SK_ID_CURR)
    bur_features = compute_agg_bureau_features(df_bur_)

    return bur_features


# ------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- INSTAL --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def cleaning_df_instal(df_instal: pd.DataFrame) -> pd.DataFrame:
    df_instal["AMT_PAYMENT"] = df_instal["AMT_PAYMENT"].fillna(
        df_instal["AMT_INSTALMENT"]
    )
    df_instal["DAYS_ENTRY_PAYMENT"] = df_instal["DAYS_ENTRY_PAYMENT"].fillna(
        df_instal["DAYS_INSTALMENT"]
    )
    return df_instal


def compute_first_agg_instal(df_instal: pd.DataFrame) -> pd.DataFrame:
    # We will now proceed to the first aggregation, this one will take place over SK_ID_PREV
    # We first compute the differences, and create two new features within our dataset
    df_instal["DIFF_AMT_INSTALMENT_PAYMENT"] = (
        df_instal["AMT_INSTALMENT"] - df_instal["AMT_PAYMENT"]
    )
    df_instal["DIFF_DAYS_INSTALMENT_PAYMENT"] = (
        df_instal["DAYS_INSTALMENT"] - df_instal["DAYS_ENTRY_PAYMENT"]
    )

    # We'll start with non-temporal features
    agg_dict_all = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "NUM_INSTALMENT_NUMBER": ["max", "mean"],  # mean added
        "DAYS_ENTRY_PAYMENT": ["max", "min"],
        "AMT_PAYMENT": ["max", "mean", "std"],  # max, std added
        "AMT_INSTALMENT": ["max", "mean"],  # added
        "DAYS_ENTRY_PAYMENT": ["max", "min", "mean"],  # added
    }
    features_all = df_instal.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg(agg_dict_all)
    features_all.columns = [
        "_".join(col).upper() for col in features_all.columns.values
    ]

    # Now we can proceed with the temporal ones
    time_windows = [
        ("EVER", df_instal),
        ("LAST_YEAR", df_instal[df_instal["DAYS_ENTRY_PAYMENT"].between(-365, -92)]),
        ("LAST_3_MONTHS", df_instal[df_instal["DAYS_ENTRY_PAYMENT"] > -92]),
    ]
    diff_features = []
    for window_name, window_df in time_windows:
        agg_dict_diff = {
            "DIFF_AMT_INSTALMENT_PAYMENT": ["sum", "mean"],  # mean added
            "DIFF_DAYS_INSTALMENT_PAYMENT": ["sum", "mean"],  # mean added
        }
        window_features = window_df.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg(
            agg_dict_diff
        )
        window_features.columns = [
            f"{col[0]}_{window_name.upper()}_{col[1].upper()}"
            for col in window_features.columns.values
        ]
        diff_features.append(window_features)

    # We can combine all the features, reset the index, fill the missing values for the temporal features by 0 and drop the SK_ID_PREV
    first_agg_features = (
        features_all.join(diff_features)
        .fillna(0)
        .reset_index()
        .drop("SK_ID_PREV", axis=1)
    )

    return first_agg_features


def compute_second_agg_instal(first_agg_features: pd.DataFrame) -> pd.DataFrame:
    # Define aggregation operations for each group of features
    agg_ops = {
        # Temporal features (diff amount and diff days for each period)
        "DIFF_AMT_INSTALMENT_PAYMENT_EVER_SUM": ["mean", "sum", "max"],  # max added
        "DIFF_DAYS_INSTALMENT_PAYMENT_EVER_SUM": ["mean", "sum", "max"],  # max added
        "DIFF_AMT_INSTALMENT_PAYMENT_LAST_YEAR_SUM": [
            "mean",
            "sum",
            "max",
        ],  # max added
        "DIFF_DAYS_INSTALMENT_PAYMENT_LAST_YEAR_SUM": [
            "mean",
            "sum",
            "max",
        ],  # max added
        "DIFF_AMT_INSTALMENT_PAYMENT_LAST_3_MONTHS_SUM": [
            "mean",
            "sum",
            "max",
        ],  # max added
        "DIFF_DAYS_INSTALMENT_PAYMENT_LAST_3_MONTHS_SUM": [
            "mean",
            "sum",
            "max",
        ],  # max added
        # Number of changes in installment version
        "NUM_INSTALMENT_VERSION_NUNIQUE": ["mean", "sum", "max"],  # max added
        # Number of installments
        "NUM_INSTALMENT_NUMBER_MAX": ["mean", "sum", "max"],
        # Oldest and most recent installments
        "DAYS_ENTRY_PAYMENT_MAX": ["mean", "max", "min", "std"],  # std added
        "DAYS_ENTRY_PAYMENT_MIN": ["mean", "max", "min", "std"],  # std added
        # Average and max payment
        "AMT_PAYMENT_MEAN": ["mean", "max", "std"],  # std added
        #'AMT_PAYMENT_MAX': ['mean', 'max', 'std'] #std added
    }

    # Perform the second aggregation
    second_agg_features = first_agg_features.groupby("SK_ID_CURR").agg(agg_ops)
    second_agg_features.columns = [
        "INSTAL_" + "_".join(col).upper() for col in second_agg_features.columns.values
    ]

    return second_agg_features.reset_index()


def compute_features_instal(df_instal: pd.DataFrame) -> pd.DataFrame:
    df_instal_ = df_instal.copy()

    # Clean the dataset
    df_instal_ = cleaning_df_instal(df_instal_)

    # Compute the first agg features (over SK_ID_PREV)
    first_agg = compute_first_agg_instal(df_instal_)

    # Compute the second agg features (over SK_ID_CURR)
    second_agg = compute_second_agg_instal(first_agg)

    # Reset the index to be SK_ID_CURR
    second_agg.set_index("SK_ID_CURR", inplace=True)

    return second_agg


# ------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------- POS CASH ------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def compute_first_agg_pos_cash(df_pos_cash: pd.DataFrame) -> pd.DataFrame:
    # We will now proceed to the first aggregation, this one will take place over SK_ID_PREV
    # We first compute the ratio
    df_pos_cash["RATIO_DPD_DEF_DPD"] = df_pos_cash["SK_DPD_DEF"] / (
        df_pos_cash["SK_DPD"] + 0.0001
    )

    # We then order the dataset using MONTHS_BALANCE
    df_pos_cash.sort_values(by="MONTHS_BALANCE", ascending=False, inplace=True)

    # We compute LAST_STATUS_POS in the most efficient way (first much slower)
    last_status_pos = (
        df_pos_cash.groupby(["SK_ID_PREV", "SK_ID_CURR"])["NAME_CONTRACT_STATUS"]
        .nth(0)
        .rename("LAST_STATUS_POS")
    )

    last_status_pos = (
        pd.concat(
            [
                df_pos_cash.loc[last_status_pos.index][["SK_ID_PREV", "SK_ID_CURR"]],
                last_status_pos,
            ],
            axis=1,
        )
        .sort_values(by=["SK_ID_PREV", "SK_ID_CURR"])
        .set_index(["SK_ID_PREV", "SK_ID_CURR"])
    )

    # We'll start with non-temporal features
    agg_dict_all = {
        "CNT_INSTALMENT_FUTURE": ["min", "sum"],  # sum added
        "CNT_INSTALMENT": ["max", "sum"],  # sum added
    }
    features_all = df_pos_cash.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg(agg_dict_all)
    features_all.columns = [
        "_".join(col).upper() for col in features_all.columns.values
    ]
    features_all = pd.concat([last_status_pos, features_all], axis=1)

    # Now we can proceed with the temporal ones
    time_windows = [
        ("EVER", df_pos_cash),
        ("LAST_YEAR", df_pos_cash[df_pos_cash["MONTHS_BALANCE"].between(-12, -1)]),
        ("LAST_3_MONTHS", df_pos_cash[df_pos_cash["MONTHS_BALANCE"] > -3]),
    ]
    diff_features = []
    all_window_columns = []
    for window_name, window_df in time_windows:
        agg_dict_diff = {
            "SK_DPD": ["max"],
            "SK_DPD_DEF": ["max"],
            "RATIO_DPD_DEF_DPD": ["max", "mean"],  # mean added
        }
        window_features = window_df.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg(
            agg_dict_diff
        )
        window_features.columns = [
            f"{col[0]}_{window_name.upper()}_{col[1].upper()}"
            for col in window_features.columns.values
        ]
        diff_features.append(window_features)
        all_window_columns += list(window_features.columns.values)

    # We can combine all the features, reset the index, fill the missing values for the temporal features by 0 and drop the SK_ID_PREV
    first_agg_features = (
        features_all.join(diff_features).reset_index().drop("SK_ID_PREV", axis=1)
    )
    first_agg_features[all_window_columns] = first_agg_features[
        all_window_columns
    ].fillna(0)

    return first_agg_features


def compute_second_agg_pos_cash(first_agg_features: pd.DataFrame) -> pd.DataFrame:
    dum_status = pd.get_dummies(
        first_agg_features["LAST_STATUS_POS"], dtype=np.int8, prefix="LAST_STATUS_POS"
    )

    first_agg_features = pd.concat([dum_status, first_agg_features], axis=1)
    first_agg_features.drop("LAST_STATUS_POS", axis=1, inplace=True)

    status_columns = [
        col for col in first_agg_features.columns if col.startswith("LAST_STATUS")
    ]
    temporal_columns = [
        col
        for col in first_agg_features.columns
        if any(p in col for p in ["EVER", "LAST_YEAR", "LAST_3_MONTHS"])
    ]

    # Create the aggregation dictionaries
    agg_dict = {col: "sum" for col in status_columns}
    agg_temp = {col: ["sum", "mean"] for col in temporal_columns}  # sum added
    agg_misc = {
        "CNT_INSTALMENT_FUTURE_MIN": ["mean", "sum", "max"],  # max added
        "CNT_INSTALMENT_MAX": ["mean", "sum", "max"],  # max added
    }

    agg_dict.update(agg_temp)
    agg_dict.update(agg_misc)

    # Perform the second aggregation
    second_agg_features = first_agg_features.groupby("SK_ID_CURR").agg(agg_dict)
    second_agg_features.columns = [
        "POS_CASH_" + "_".join(col).upper()
        for col in second_agg_features.columns.values
    ]

    return second_agg_features.reset_index()


def compute_features_pos_cash(df_pos_cash: pd.DataFrame) -> pd.DataFrame:
    df_pos_cash_ = df_pos_cash.copy()

    # Compute the first agg features (over SK_ID_PREV)
    first_agg = compute_first_agg_pos_cash(df_pos_cash_)

    # Compute the second agg features (over SK_ID_CURR)
    second_agg = compute_second_agg_pos_cash(first_agg)

    # Reset the index to be SK_ID_CURR
    second_agg.set_index("SK_ID_CURR", inplace=True)

    return second_agg
