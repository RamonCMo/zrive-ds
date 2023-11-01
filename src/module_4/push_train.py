import numpy as np
import pandas as pd
import json
import joblib
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

nuemric = [
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
    "normalised_price",
    "discount_pct",
    "global_popularity",
]
ordinal = [
    "user_order_seq",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "days_since_purchase_product_type",
]
binary = [
    "outcome",
    "ordered_before",
    "abandoned_before",
    "active_snoozed",
    "set_as_regular",
]
categrorical = ["product_type", "vendor"]
ides = ["order_id", "user_id", "variant_id"]
date = ["created_at", "order_date"]


def filter_orders(data: pd.DataFrame) -> pd.DataFrame:
    order_counts = (
        data.groupby("order_id")["outcome"].sum().reset_index(name="products_bought")
    )
    filtered_orders = order_counts[order_counts["products_bought"] >= 5]
    final_dataframe = data[data["order_id"].isin(filtered_orders["order_id"])]
    return final_dataframe


def data_load(
    path: str,
    nuemric: list,
    ordinal: list,
    binary: list,
    categrorical: list,
    ides: list,
    date: list,
) -> pd.DataFrame:
    cols = nuemric + ordinal + binary + categrorical + ides + date
    try:
        df = pd.read_csv(path, delimiter=",")
        if not set(cols).issubset(list(df.columns)):
            raise Exception("El DataFrame no contiene todas las columnas esperadas.")
        if df.empty:
            raise Exception("El DataFrame está vacío.")

        df[ordinal] = df[ordinal].astype(np.int8)
        df[binary] = df[binary].astype(np.int8)
        df[nuemric] = df[nuemric].astype(np.float64)
        df[date[0]] = pd.to_datetime(df[date[0]])
        df[date[1]] = pd.to_datetime(df[date[1]])
        df = filter_orders(data=df)

        if df.isnull().values.any():
            raise Exception("El DataFrame contiene valores nulos que hay que imputar.")
        return df

    except FileNotFoundError:
        print(f"Error: El archivo CSV no se encuentra en el path especificado: {path}")
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío.")
    except Exception as e:
        print(f"Error: {str(e)}")
    return None


def discretize_variable(df: pd.DataFrame, labels: list, variable: str) -> pd.DataFrame:
    discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    df[variable] = discretizer.fit_transform(df[[variable]])
    df[variable] = df[variable].apply(lambda x: labels[int(x)])
    return df


def encoder(df: pd.DataFrame, var: str) -> pd.DataFrame:
    encoder = OneHotEncoder(
        sparse=False,
        dtype=int,
        categories="auto",
        handle_unknown="ignore",
        max_categories=10,
    )
    product_type_encoded = encoder.fit_transform(df[[var]])
    categories = encoder.get_feature_names_out()
    product_type_df = pd.DataFrame(product_type_encoded, columns=categories)
    product_type_df.index = df.index
    encoded_df = pd.concat([df.drop(var, axis=1), product_type_df], axis=1)
    return encoded_df


def frequency_encoding(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    frequencies = dataframe[column_name].value_counts(normalize=True)
    dataframe[column_name + "_encoded"] = dataframe[column_name].map(frequencies)
    dataframe = dataframe.drop([column_name], axis=1)
    return dataframe


def scaler(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = frequency_encoding(dataframe, "vendor")
    sc = StandardScaler()
    dataframe = sc.fit_transform(dataframe)
    return dataframe


def preproccesing(df: pd.DataFrame, date_split: str) -> (pd.DataFrame, pd.DataFrame):
    df = df.drop(
        [
            "count_adults",
            "count_children",
            "count_pets",
            "std_days_to_buy_variant_id",
            "std_days_to_buy_product_type",
            "created_at",
        ]
        + ides,
        axis=1,
    )
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day_of_year"] = df["order_date"].dt.day_of_year
    df_encoded = encoder(df=df, var="product_type")
    labels = ["Weekly", "Monthly", "Yearly", "Long_dur"]
    discretized_df = discretize_variable(
        df=df_encoded, labels=labels, variable="avg_days_to_buy_variant_id"
    )
    df_features = encoder(df=discretized_df, var="avg_days_to_buy_variant_id")
    df_train = df_features[df_features["order_date"] < date_split].drop(
        ["order_date"], axis=1
    )
    X = scaler(df_train.drop(columns="outcome"))
    y = df_train["outcome"]
    return X, y


def handler_fit(event: dict, _) -> dict:
    if not isinstance(event["model_parametrisation"], dict):
        raise ValueError("Parameter Grid is not a Dictionary.")

    X, y = preproccesing(
        data_load(
            path="../../data/feature_frame.csv",
            nuemric=nuemric,
            ordinal=ordinal,
            binary=binary,
            categrorical=categrorical,
            ides=ides,
            date=date,
        )
    )

    model = CalibratedClassifierCV(
        RandomForestClassifier(**event["model_parametrisation"]),
        **event["calibration_parametrisation"],
    )

    model_fit = model.fit(X, y)
    today_date = datetime.now().strftime("%Y_%m_%d")
    model_path = f"src/module_4/push_{today_date}.pkl"
    joblib.dump(model_fit, model_path)

    return {"statusCode": "200", "body": json.dumps({"model_path": [model_path]})}
