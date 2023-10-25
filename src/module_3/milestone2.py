import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)


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
        df = pd.read_csv(path)
        if not set(cols).issubset(df.columns):
            raise Exception("El DataFrame no contiene todas las columnas esperadas.")
        if df.empty:
            raise Exception("El DataFrame está vacío.")

        df[ordinal] = df[ordinal].astype(np.int8)
        df[binary] = df[[binary]].astype(np.int8)
        df[nuemric] = df[nuemric].astype(np.float64)
        df[date] = pd.to_datetime(df[date])
        df = filter_orders(data=df)
        df = df.drop(ides, axis=1)

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


def preproccesig(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df = df.drop(
        [
            "count_adults",
            "count_children",
            "count_pets",
            "std_days_to_buy_variant_id",
            "std_days_to_buy_product_type",
            "created_at",
        ],
        axis=1,
    )
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["day_of_year"] = df["order_date"].dt.day_of_year
    df = df.drop(["order_date"], axis=1)
    df_encoded = encoder(df=df, var="product_type")
    labels = ["Weekly", "Monthly", "Yearly", "Long_dur"]
    discretized_df = discretize_variable(
        df=df_encoded, labels=labels, variable="avg_days_to_buy_variant_id"
    )
    df_features = encoder(df=discretized_df, var="avg_days_to_buy_variant_id")
    X = df_features.drop(columns="outcome")
    y = df_features["outcome"]
    return X, y


def model_training(X_train: pd.DataFrame, y_train: pd.Series) -> (GridSearchCV, dict):
    pipe = Pipeline(
        [("classifier", LogisticRegression(random_state=42, solver="saga"))]
    )
    param_grid = {
        "classifier__penalty": ["l1", "l2", None],
        "classifier__C": [1e-6, 0.1, 1, 10],
    }
    grid_search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    resultados = grid_search.cv_results_
    return grid_search, resultados


def models_viewer(
    X_val: pd.DataFrame, y_val: pd.Series, grid_search: GridSearchCV, resultados: dict
) -> None:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC")
    plt.subplot(1, 2, 2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas de Precisión-Recall")

    for i in range(len(resultados["params"])):
        modelo = grid_search.best_estimator_.set_params(**resultados["params"][i])
        y_pred_prob3 = modelo.predict_proba(X_val)[:, 1]

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob3)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, lw=0.5, label=f"Modelo {i+1} (AUC = {roc_auc:.2f})")

        # Curva de Recall-Precisión
        precision, recall, _ = precision_recall_curve(y_val, y_pred_prob3)
        pr_auc = auc(recall, precision)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, lw=0.5, label=f"Modelo {i+1} (AUC = {pr_auc:.2f}")
    plt.subplot(1, 2, 1)
    plt.legend(loc="lower right")
    plt.subplot(1, 2, 2)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def final_model(path_name: str, best_params: dict, X: pd.DataFrame, y: pd.Series) -> None:
    pipe = Pipeline(
        [("classifier", LogisticRegression(random_state=42, solver="saga"))]
    )
    grid_search = GridSearchCV(pipe, best_params, cv=5, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X, y)
    joblib.dump(grid_search, f"{path_name}.pkl")


def filter_orders(data: pd.DataFrame) -> pd.DataFrame:
    order_counts = (
        data.groupby("order_id")["outcome"].sum().reset_index(name="products_bought")
    )
    filtered_orders = order_counts[order_counts["products_bought"] >= 5]
    final_dataframe = data[data["order_id"].isin(filtered_orders["order_id"])]
    return final_dataframe


def count_null_values(df: pd.DataFrame) -> dict:
    null_counts = df.isna().sum()
    null_dict = null_counts.to_dict()
    return null_dict


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
    product_type_df = pd.DataFrame(
        product_type_encoded, columns=categories
    )  # .reset_index(drop=True, inplace=True)
    product_type_df.index = df.index
    encoded_df = pd.concat([df.drop(var, axis=1), product_type_df], axis=1)
    return encoded_df


def discretize_variable(df: pd.DataFrame, labels: list, variable: str) -> pd.DataFrame:
    discretizer = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
    df[variable] = discretizer.fit_transform(df[[variable]])
    df[variable] = df[variable].apply(lambda x: labels[int(x)])
    return df


def frequency_encoding(dataframe, column_name):
    frequencies = dataframe[column_name].value_counts(normalize=True)
    dataframe[column_name + "_encoded"] = dataframe[column_name].map(frequencies)
    dataframe = dataframe.drop([column_name], axis=1)
    return dataframe


def scaler(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    X_train = frequency_encoding(train, "vendor")
    X_val = frequency_encoding(validation, "vendor")
    X_test = frequency_encoding(test, "vendor")
    sc = StandardScaler()
    X_train = sc.fit_transform(train)
    X_test = sc.transform(validation)
    X_val = sc.transform(test)
    return X_train, X_test, X_val


def main() -> None:
    PATH = "../../data/"
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

    data = data_load(
        path=PATH,
        nuemric=nuemric,
        ordinal=ordinal,
        binary=binary,
        categrorical=categrorical,
        ides=ides,
        date=date,
    )
    X, y = preproccesig(df=data)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.4, stratify=y_temp, random_state=42
    )
    X_train, X_test, X_val = scaler(train=X_train, validation=X_val, test=X_test)
    grid_search, resultados = model_training(X_train=X_train, y_train=y_train)
    models_viewer(
        X_val=X_val, y_val=y_val, grid_search=grid_search, resultados=resultados
    )
    final_model(
        path_name="models/final_model_file", best_params=grid_search.best_params_, X=X
    )


if __name__ == "__main__":
    main()
