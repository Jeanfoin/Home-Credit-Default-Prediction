import pandas as pd
import numpy as np
from typing import Callable
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
        StandardScaler,
        PolynomialFeatures,
        TargetEncoder
)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


def compute_features_credit_card(df_credit: pd.DataFrame) -> pd.DataFrame:
    """This function aims at computing the relevant features from the credit
    card balance dataset
    """
    df_credit_ = df_credit.copy()

    df_credit_["PAYMENT_RECEIVABLE_RATIO"] = df_credit_["AMT_TOTAL_RECEIVABLE"] / (
        df_credit_["AMT_PAYMENT_TOTAL_CURRENT"] + 0.0001
    )

    # filtered dataset
    filtered = df_credit_[df_credit_["MONTHS_BALANCE"].between(-12, -1)]

    # return filtered
    # first aggregation (SK_ID_PREV)
    first_agg = filtered.groupby(["SK_ID_PREV", "SK_ID_CURR"])[
        "PAYMENT_RECEIVABLE_RATIO"
    ].sum()
    first_agg = first_agg.reset_index().drop("SK_ID_PREV", axis=1).fillna(0)
    # return first_agg
    # second aggregation (SK_ID_CURR)
    credit_card_features = first_agg.groupby("SK_ID_CURR").sum()

    return credit_card_features.add_prefix("CREDIT_BALANCE_").add_suffix(
        "_LAST_YEAR_SUM_SUM"
    )


def compute_features_previous(df_previous: pd.DataFrame) -> pd.DataFrame:
    """This function aims at computing the relevant features from the previous
    application dataset
    """
    df_previous_ = df_previous.copy()

    df_previous_.loc[
        df_previous_["DAYS_FIRST_DUE"] == 365243.0, "DAYS_FIRST_DUE"
    ] = np.nan
    df_previous_.loc[
        df_previous_["DAYS_TERMINATION"] == 365243.0, "DAYS_TERMINATION"
    ] = np.nan

    df_previous_["AMT_MISSING_CREDIT"] = (
        df_previous_["AMT_APPLICATION"] - df_previous_["AMT_CREDIT"]
    )
    df_previous_["AMT_CREDIT_GOODS_DIFF"] = (
        df_previous_["AMT_GOODS_PRICE"] - df_previous_["AMT_CREDIT"]
    )
    df_previous_["DIFF_CREDIT_DOWN_PAYMENT"] = (
        df_previous_["AMT_CREDIT"] - df_previous_["AMT_DOWN_PAYMENT"]
    )
    df_previous_["AMT_INTEREST_LOAN"] = (
        df_previous_["CNT_PAYMENT"] * df_previous_["AMT_ANNUITY"]
        - df_previous_["AMT_CREDIT"]
    )
    df_previous_["LOANS_DURATION"] = (
        df_previous_["DAYS_TERMINATION"] - df_previous_["DAYS_FIRST_DUE"]
    )

    agg_dict = {
        "AMT_ANNUITY": ["sum"],
        "AMT_MISSING_CREDIT": ["mean"],
        "AMT_CREDIT_GOODS_DIFF": ["sum", "max"],
        "DIFF_CREDIT_DOWN_PAYMENT": ["sum", "mean"],
        "AMT_INTEREST_LOAN": ["mean"],
        "DAYS_TERMINATION": ["mean"],
        "LOANS_DURATION": ["max", "min"],
    }

    previous_features = df_previous_.groupby("SK_ID_CURR").agg(agg_dict)
    previous_features.columns = [
        "PREVIOUS_" + "_".join(col).upper() for col in previous_features.columns.values
    ]

    return previous_features


def compute_features_bureau(df_bureau: pd.DataFrame) -> pd.DataFrame:
    """This function aims at computing the relevant features from the credit
    bureau dataset
    """
    df_bureau_ = df_bureau.copy()

    df_bureau_.loc[
        np.abs(df_bureau_["DAYS_CREDIT_ENDDATE"]) > 15000, "DAYS_CREDIT_ENDDATE"
    ] = np.nan
    df_bureau_.loc[
        np.abs(df_bureau_["DAYS_ENDDATE_FACT"]) > 15000, "DAYS_ENDDATE_FACT"
    ] = np.nan
    df_bureau_.loc[
        np.abs(df_bureau_["AMT_CREDIT_SUM"]) > 1e8, "AMT_CREDIT_SUM"
    ] = np.nan
    df_bureau_.loc[
        np.abs(df_bureau_["AMT_CREDIT_SUM_DEBT"]) > 5e7, "AMT_CREDIT_SUM_DEBT"
    ] = np.nan

    df_bureau_["CREDIT_DURATION"] = (
        df_bureau_["DAYS_CREDIT_ENDDATE"] - df_bureau_["DAYS_CREDIT"]
    )
    df_bureau_["EFFECTIVE_CREDIT_DURATION"] = (
        df_bureau_["DAYS_ENDDATE_FACT"] - df_bureau_["DAYS_CREDIT"]
    )
    df_bureau_["RATIO_CURRENT_DEBT_CREDIT"] = df_bureau_["AMT_CREDIT_SUM_DEBT"] / (
        df_bureau_["AMT_CREDIT_SUM"] + 0.0001
    )
    df_bureau_["RATIO_AMT_CREDIT_LIMIT"] = df_bureau_["AMT_CREDIT_SUM"] / (
        df_bureau_["AMT_CREDIT_SUM_LIMIT"] + 0.0001
    )

    filtered = df_bureau_[df_bureau["CREDIT_ACTIVE"] == "Active"]

    agg_dict = {
        "DAYS_CREDIT": ["min", "max"],
        "AMT_CREDIT_SUM": ["max", "sum"],
        "EFFECTIVE_CREDIT_DURATION": ["mean", "min", "max"],
        "RATIO_CURRENT_DEBT_CREDIT": ["mean", "max"],
        "RATIO_AMT_CREDIT_LIMIT": ["mean", "max"],
    }

    bureau_features = filtered.groupby("SK_ID_CURR").agg(agg_dict)
    bureau_features.columns = [
        "BUREAU_ACTIVE_" + "_".join(col).upper()
        for col in bureau_features.columns.values
    ]

    return bureau_features.fillna(0)


def compute_features_instalment(df_instalment: pd.DataFrame) -> pd.DataFrame:
    df_instalment_ = df_instalment.copy()

    df_instalment_["AMT_PAYMENT"] = df_instalment_["AMT_PAYMENT"].fillna(
        df_instalment_["AMT_INSTALMENT"]
    )
    df_instalment_["DAYS_ENTRY_PAYMENT"] = df_instalment_["DAYS_ENTRY_PAYMENT"].fillna(
        df_instalment_["DAYS_INSTALMENT"]
    )

    df_instalment_["DIFF_DAYS_INSTALMENT_PAYMENT"] = (
        df_instalment_["DAYS_INSTALMENT"] - df_instalment_["DAYS_ENTRY_PAYMENT"]
    )

    first_agg_dict = {
        "NUM_INSTALMENT_NUMBER": ["max"],
        "DAYS_ENTRY_PAYMENT": ["max", "min"],
        "AMT_PAYMENT": ["mean"],
    }

    features_first_agg = df_instalment_.groupby(["SK_ID_PREV", "SK_ID_CURR"]).agg(
        first_agg_dict
    )
    features_first_agg.columns = [
        "_".join(col).upper() for col in features_first_agg.columns.values
    ]

    diff_days_instalment_payment_ever = (
        df_instalment_.groupby(["SK_ID_PREV", "SK_ID_CURR"])[
            "DIFF_DAYS_INSTALMENT_PAYMENT"
        ]
        .sum()
        .rename("DIFF_DAYS_INSTALMENT_PAYMENT_EVER_SUM")
    )
    diff_days_instalment_payment_ever.rename(
        "DIFF_DAYS_INSTALMENT_PAYMENT_EVER_SUM", inplace=True
    )

    last_year = df_instalment_[
        df_instalment_["DIFF_DAYS_INSTALMENT_PAYMENT"].between(-365, -92)
    ]
    diff_days_instalment_payment_last_year = (
        last_year.groupby(["SK_ID_PREV", "SK_ID_CURR"])[
            "DIFF_DAYS_INSTALMENT_PAYMENT"
        ].sum()
    ).rename("DIFF_DAYS_INSTALMENT_PAYMENT_LAST_YEAR_SUM")

    last_3_months = df_instalment_[df_instalment_["DIFF_DAYS_INSTALMENT_PAYMENT"] > -92]
    diff_days_instalment_payment_last_3_months = (
        last_3_months.groupby(["SK_ID_PREV", "SK_ID_CURR"])[
            "DIFF_DAYS_INSTALMENT_PAYMENT"
        ]
        .sum()
        .rename("DIFF_DAYS_INSTALMENT_PAYMENT_LAST_3_MONTHS_SUM")
    )

    window_features = [
        diff_days_instalment_payment_ever,
        diff_days_instalment_payment_last_year,
        diff_days_instalment_payment_last_3_months,
    ]

    first_agg_features = (
        features_first_agg.join(window_features)
        .fillna(0)
        .reset_index()
        .drop("SK_ID_PREV", axis=1)
    )

    second_agg_dict = {
        "DIFF_DAYS_INSTALMENT_PAYMENT_EVER_SUM": ["mean", "sum", "max"],
        "DIFF_DAYS_INSTALMENT_PAYMENT_LAST_YEAR_SUM": ["mean"],
        "DIFF_DAYS_INSTALMENT_PAYMENT_LAST_3_MONTHS_SUM": ["mean"],
        "NUM_INSTALMENT_NUMBER_MAX": ["mean"],
        "DAYS_ENTRY_PAYMENT_MAX": ["max"],
        "DAYS_ENTRY_PAYMENT_MIN": ["std"],
        "AMT_PAYMENT_MEAN": ["mean", "max"],
    }

    instalment_features = first_agg_features.groupby("SK_ID_CURR").agg(second_agg_dict)
    instalment_features.columns = [
        "INSTAL_" + "_".join(col).upper() for col in instalment_features.columns.values
    ]

    return instalment_features


def compute_features_pos_cash(df_pos_cash: pd.DataFrame) -> pd.DataFrame:
    df_pos_cash_ = df_pos_cash.copy()

    cnt_instalment_max = (
        df_pos_cash_.groupby(["SK_ID_PREV", "SK_ID_CURR"])["CNT_INSTALMENT"]
        .max()
        .rename("CNT_INSTALMENT_MAX")
        .reset_index()
        .drop("SK_ID_PREV", axis=1)
        .fillna(0)
    )

    pos_cash_features = cnt_instalment_max.groupby("SK_ID_CURR").agg(["mean", "sum"])
    pos_cash_features.columns = [
        "POS_CASH_" + "_".join(col).upper() for col in pos_cash_features.columns.values
    ]

    return pos_cash_features


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


def load_additional_df(
    sk_id_curr: pd.Series, filename: str, dir_loc: str = "../add_data/"
) -> pd.DataFrame:
    tmp_ds = ds.dataset(dir_loc + filename)
    tmp_table = tmp_ds.to_table(filter=(ds.field("SK_ID_CURR").isin(sk_id_curr)))
    df = tmp_table.to_pandas()

    return df


def join_with_app(app_df: pd.DataFrame, additional_df: pd.DataFrame) -> pd.DataFrame:
    sk_id_curr = pd.DataFrame(index=app_df.index)
    pre_joined = sk_id_curr.join(additional_df, how="left").fillna(0)

    joined = app_df.join(pre_joined, how="left")
    return joined


def load_compute_and_join_with_app(
    app_df: pd.DataFrame,
    filename: str,
    compute_func: Callable[[pd.DataFrame], pd.DataFrame],
    dir_loc: str = "../add_data/",
) -> pd.DataFrame:
    sk_id_curr = app_df.index
    # Load previous application data set
    additional_df = load_additional_df(sk_id_curr, filename, dir_loc)
    # Compute the features
    additional_features = compute_func(additional_df)
    # Delete and gets memory back
    del additional_df
    gc.collect()
    # Join the dataset to the main one
    augmented_df = join_with_app(app_df, additional_features)
    # return additional_features
    return augmented_df


def get_full_dataframe(
    app_df: pd.DataFrame, dir: str = "../add_data/"
) -> pd.DataFrame:
    full_features = load_compute_and_join_with_app(
        app_df, "credit_card_balance.parquet", compute_features_credit_card,
        dir
    )

    full_features = load_compute_and_join_with_app(
        full_features, "previous_application.parquet",
        compute_features_previous,
        dir
    )

    full_features = load_compute_and_join_with_app(
        full_features, "bureau.parquet", compute_features_bureau,
        dir
    )

    full_features = load_compute_and_join_with_app(
        full_features, "installments_payments.parquet",
        compute_features_instalment,
        dir
    )

    full_features = load_compute_and_join_with_app(
        full_features, "POS_CASH_balance.parquet", compute_features_pos_cash,
        dir
    )

    return full_features.set_index("SK_ID_CURR")


class JoinDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, add_dir):
        self.add_dir = add_dir

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()

        X_ = get_full_dataframe(X_, dir=self.add_dir)
        return X_


# ----------------------------------------------------------------------------
# -------------------- Anomalies remover and dropper -------------------------
# ----------------------------------------------------------------------------


COLUMNS_TO_DROP = [
    "FLAG_DOCUMENT_11",
    "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_9",
    "FLAG_DOCUMENT_14",
    "FLAG_CONT_MOBILE",
    "FLAG_DOCUMENT_15",
    "FLAG_DOCUMENT_19",
    "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
    "FLAG_DOCUMENT_17",
    "FLAG_DOCUMENT_7",
    "FLAG_DOCUMENT_2",
    "FLAG_DOCUMENT_4",
    "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_12",
    "FLAG_MOBIL",
    "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE"
]


class RemoveAnomalieRescaleAndDrop(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_.loc[X_["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan
        X_["DAYS_EMPLOYED"] = -X_["DAYS_EMPLOYED"] / 365
        X_["DAYS_BIRTH"] = -X_["DAYS_BIRTH"] / 365
        X_["DAYS_REGISTRATION"] = -X_["DAYS_REGISTRATION"] / 365
        X_["DAYS_ID_PUBLISH"] = -X_["DAYS_ID_PUBLISH"] / 365
        X_.drop(COLUMNS_TO_DROP, axis=1, inplace=True)
        return X_


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

HOUSING_BLOCK_COLS = [
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "YEARS_BUILD_AVG",
    "COMMONAREA_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "FLOORSMIN_AVG",
    "LANDAREA_AVG",
    "LIVINGAPARTMENTS_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAPARTMENTS_AVG",
    "NONLIVINGAREA_AVG",
    "APARTMENTS_MODE",
    "BASEMENTAREA_MODE",
    "YEARS_BEGINEXPLUATATION_MODE",
    "YEARS_BUILD_MODE",
    "COMMONAREA_MODE",
    "ELEVATORS_MODE",
    "ENTRANCES_MODE",
    "FLOORSMAX_MODE",
    "FLOORSMIN_MODE",
    "LANDAREA_MODE",
    "LIVINGAPARTMENTS_MODE",
    "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_MODE",
    "NONLIVINGAREA_MODE",
    "APARTMENTS_MEDI",
    "BASEMENTAREA_MEDI",
    "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_MEDI",
    "COMMONAREA_MEDI",
    "ELEVATORS_MEDI",
    "ENTRANCES_MEDI",
    "FLOORSMAX_MEDI",
    "FLOORSMIN_MEDI",
    "LANDAREA_MEDI",
    "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_MEDI",
    "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_MEDI",
    "TOTALAREA_MODE",
]

NUMERICAL_FEATURES = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE",
    "HOUR_APPR_PROCESS_START",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "DAYS_LAST_PHONE_CHANGE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="median", columns_to_impute=NUMERICAL_FEATURES + HOUSING_BLOCK_COLS):
        self.strategy = strategy
        self.columns_to_impute = columns_to_impute
        self.imputer = SimpleImputer(strategy=self.strategy)


    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns_to_impute])
        return self
    
    def transform(self, X):
        X_ = X.copy()
        
        imputed_features = self.imputer.transform(X_[self.columns_to_impute])
        X_[self.columns_to_impute] = imputed_features
        return X_


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale=HOUSING_BLOCK_COLS):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()


    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self
    
    def transform(self, X):
        X_ = X.copy()
        
        scaled_features = self.scaler.transform(X_[self.columns_to_scale])
        X_[self.columns_to_scale] = scaled_features
        return X_

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


class CustomPCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pca_features=HOUSING_BLOCK_COLS, n_components=0.9):
        self.pca_features = pca_features 
        self.n_components = n_components
        self.pca = PCA(n_components=n_components) 

    def fit(self, X, y=None):
        self.pca.fit(X[self.pca_features])
        return self
    
    def transform(self, X):
        X_ = X.copy()

        pca_result = self.pca.transform(X_[self.pca_features])
        for i in range(pca_result.shape[1]):
            X_[f'PCA_HOUSING_{i+1}'] = pca_result[f'pca{i}']

        X_.drop(self.pca_features, axis=1, inplace=True)
        
        return X_


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_["INCOME_CHILDREN_RATIO"] = X_["AMT_INCOME_TOTAL"] / (
            X_["CNT_CHILDREN"] + 0.0001
        )
        X_["CREDIT_INCOME_RATIO"] = X_["AMT_CREDIT"] / (X_["AMT_INCOME_TOTAL"] + 0.0001)
        X_["CREDIT_ANNUITY_RATIO"] = X_["AMT_CREDIT"] / (X_["AMT_ANNUITY"] + 0.0001)
        X_["ANNUITY_INCOME_RATIO"] = X_["AMT_ANNUITY"] / (
            X_["AMT_INCOME_TOTAL"] + 0.0001
        )
        X_["INCOME_ANNUITY_DIFF"] = X_["AMT_INCOME_TOTAL"] - X_["AMT_ANNUITY"]
        X_["CREDIT_GOODS_RATIO"] = X_["AMT_CREDIT"] / (X_["AMT_GOODS_PRICE"] + 0.0001)
        X_["CREDIT_GOODS_DIFF"] = X_["AMT_CREDIT"] - X_["AMT_GOODS_PRICE"] + 0.0001
        X_["GOODS_INCOME_RATIO"] = X_["AMT_GOODS_PRICE"] / (
            X_["AMT_INCOME_TOTAL"] + 0.0001
        )
        X_["AVG_EXT_SOURCE"] = (
            X_["EXT_SOURCE_1"] + X_["EXT_SOURCE_2"] + X_["EXT_SOURCE_3"]
        ) / 3
        X_["HARM_AVG_EXT_SOURCE"] = (
            X_["EXT_SOURCE_1"] * X_["EXT_SOURCE_2"] * X_["EXT_SOURCE_3"]
        ) / (X_["EXT_SOURCE_1"] + X_["EXT_SOURCE_2"] + X_["EXT_SOURCE_3"] + 0.001)
        X_["AVG_60_OBS_DEF"] = (
            X_["OBS_60_CNT_SOCIAL_CIRCLE"] + X_["DEF_60_CNT_SOCIAL_CIRCLE"]
        ) / 2
        X_["RATION_EMPLOYED_AGE"] = X_["DAYS_BIRTH"] / X_["DAYS_EMPLOYED"]

        # Combinining several datasets
        X_["RATIO_TOTAL_CREDIT_INCOME"] = (
            X_["AMT_CREDIT"]
            + X_["PREVIOUS_DIFF_CREDIT_DOWN_PAYMENT_SUM"]
            + X_["BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM"]
        ) / (X_["AMT_INCOME_TOTAL"] + 0.0001)

        return X_

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

POLY_FEATURES =  ["EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
                "CREDIT_INCOME_RATIO",
                "ANNUITY_INCOME_RATIO",
                "CREDIT_ANNUITY_RATIO"]
DEGREE = 2


class CustomPolynomialTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, poly_features=POLY_FEATURES, degree=DEGREE):
        self.poly_features = poly_features
        self.degree = degree
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=False)

    def fit(self, X, y=None):
        self.poly.fit(X[self.poly_features])
        return self

    def transform(self, X):
        X_ = X.copy()

        poly_result = self.poly.transform(X_[self.poly_features])
        for col in poly_result.columns:
            X_[col.replace(' ', '_')] = poly_result[col]

        return X_

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

CAT_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]


class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_features=CAT_FEATURES, target_type="binary"):
        self.cat_features = cat_features
        self.target_type = target_type 
        self.encoder = TargetEncoder() 
        self.features_in = []

    def fit(self, X, y):
        self.encoder.fit(X[self.cat_features], y)
        self.features_in = X.columns.values
        return self
    
    def transform(self, X):
        X_ = X.copy()
        
        encoded_features = self.encoder.transform(X_[self.cat_features])
        for col in encoded_features.columns:
            X_[col] = encoded_features[col]
        
        return X_

    def get_feature_names_out(self):
        return self.features_in
