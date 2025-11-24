import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Penguins ML App", page_icon="🐧", layout="wide")

st.title("🎈 Hello ;)")
st.write("Let's do it")


# ----------------- DATA & MODEL HELPERS -----------------

@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
    )
    return df


@st.cache_resource
def train_models(df):
    target_col = "species"
    feature_cols = [
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]

    X = df[feature_cols]
    y = df[target_col]

    # one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    # базовые модели
    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)

    # ансамбль KNN + DT
    ensemble = VotingClassifier(
        estimators=[("knn", knn), ("dt", dt)],
        voting="hard",
    )

    models = {
        "KNN": knn,
        "Decision Tree": dt,
        "Ensemble (KNN + DT)": ensemble,
    }

    metrics = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        metrics.append({"Model": name, "Accuracy": acc})
        trained_models[name] = model

    metrics_df = pd.DataFrame(metrics).sort_values(
        by="Accuracy", ascending=False
    ).reset_index(drop=True)

    best_model_name = metrics_df.iloc[0]["Model"]
    model_columns = X_encoded.columns

    return trained_models, model_columns, metrics_df, best_model_name


def encode_single_input(user_dict, model_columns):
    user_df = pd.DataFrame([user_dict])
    user_encoded = pd.get_dummies(user_df, drop_first=True)
    user_encoded = user_encoded.reindex(columns=model_columns, fill_value=0)
    return user_encoded


def encode_batch_input(df_batch, model_columns):
    batch_encoded = pd.get_dummies(df_batch, drop_first=True)
    batch_encoded = batch_encoded.reindex(columns=model_columns, fill_value=0)
    return batch_encoded


# ----------------- LOAD DATA & TRAIN MODELS -----------------

df = load_data()
models, model_columns, metrics_df, best_model_name = train_models(df)


# ----------------- SIDEBAR: SINGLE INPUT -----------------

st.sidebar.header("Input features (single penguin)")

island = st.sidebar.selectbox("Island", sorted(df["island"].unique()))

bill_length_mm = st.sidebar.slider(
    "Bill length (mm)",
    min_value=float(df.bill_length_mm.min()),
    max_value=float(df.bill_length_mm.max()),
    value=float(df.bill_length_mm.mean()),
)

bill_depth_mm = st.sidebar.slider(
    "Bill depth (mm)",
    min_value=float(df.bill_depth_mm.min()),
    max_value=float(df.bill_depth_mm.max()),
    value=float(df.bill_depth_mm.mean()),
)

flipper_length_mm = st.sidebar.slider(
    "Flipper length (mm)",
    min_value=float(df.flipper_length_mm.min()),
    max_value=float(df.flipper_length_mm.max()),
    value=float(df.flipper_length_mm.mean()),
)

body_mass_g = st.sidebar.slider(
    "Body mass (g)",
    min_value=int(df.body_mass_g.min()),
    max_value=int(df.body_mass_g.max()),
    value=int(df.body_mass_g.mean()),
)

sex = st.sidebar.selectbox("Gender", sorted(df["sex"].unique()))

user_input = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": sex,
}


# ----------------- LAYOUT TABS -----------------

tab_data, tab_viz, tab_model, tab_predict = st.tabs(
    ["Data", "Visualization", "Models & Metrics", "Prediction"]
)


# ----------------- TAB: DATA -----------------

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df)

    st.subheader("Summary statistics")
    st.dataframe(df.describe())

    with st.expander("Class distribution"):
        st.write("Species distribution:")
        st.bar_chart(df["species"].value_counts())

        st.write("Island distribution:")
        st.bar_chart(df["island"].value_counts())

        st.write("Gender distribution:")
        st.bar_chart(df["sex"].value_counts())


# ----------------- TAB: VISUALIZATION -----------------

with tab_viz:
    st.subheader("Scatter plot")
    st.write("Body mass vs bill length, colored by species")

    st.scatter_chart(
        data=df,
        x="bill_length_mm",
        y="body_mass_g",
        color="species",
    )

    st.subheader("Custom feature scatter")

    numeric_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
    x_feat = st.selectbox("X-axis", numeric_cols, index=0)
    y_feat = st.selectbox("Y-axis", numeric_cols, index=1)

    chart = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=x_feat,
            y=y_feat,
            color="species",
            tooltip=["species", "island", "sex", x_feat, y_feat],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# ----------------- TAB: MODELS & METRICS -----------------

with tab_model:
    st.subheader("Models and their accuracy")

    st.dataframe(
        metrics_df.style.format({"Accuracy": "{:.3f}"})
    )

    best_row = metrics_df.iloc[0]
    st.success(
        f"Best model: {best_row['Model']} (Accuracy = {best_row['Accuracy']:.3f})"
    )

    st.subheader("Encoded feature columns")
    st.write(list(model_columns))


# ----------------- TAB: PREDICTION -----------------

with tab_predict:
    st.subheader("Prediction")

    # выбор модели для предсказания
    selected_model_name = st.selectbox(
        "Choose model for prediction",
        options=list(models.keys()),
        index=list(models.keys()).index(best_model_name),
    )

    selected_model = models[selected_model_name]

    st.write("Your single penguin input:")
    st.json(user_input)

    mode = st.radio(
        "Prediction mode",
        ["Single penguin (from sidebar)", "Batch prediction from CSV"],
        horizontal=True,
    )

    # ---- Single prediction ----
    if mode == "Single penguin (from sidebar)":
        user_encoded = encode_single_input(user_input, model_columns)

        if st.button("Predict species"):
            pred = selected_model.predict(user_encoded)[0]
            # KNN и ансамбль поддерживают predict_proba, DecisionTree тоже
            proba = selected_model.predict_proba(user_encoded)[0]

            st.success(f"Model: {selected_model_name}")
            st.success(f"Predicted species: {pred}")

            proba_df = pd.DataFrame(
                [proba],
                columns=selected_model.classes_,
            )
            st.write("Class probabilities:")
            st.dataframe(proba_df)

            proba_long = proba_df.T.reset_index()
            proba_long.columns = ["species", "probability"]

            prob_chart = (
                alt.Chart(proba_long)
                .mark_bar()
                .encode(
                    x="species",
                    y="probability",
                    tooltip=["species", "probability"],
                )
            )
            st.altair_chart(prob_chart, use_container_width=True)

    # ---- Batch prediction ----
    else:
        st.write("Upload a CSV file with columns:")
        st.code(
            "island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex"
        )

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write("First rows of your file:")
            st.dataframe(batch_df.head())

            if st.button("Run batch prediction"):
                batch_encoded = encode_batch_input(batch_df, model_columns)
                preds = selected_model.predict(batch_encoded)

                result_df = batch_df.copy()
                result_df["prediction"] = preds

                st.success(f"Predictions completed using {selected_model_name}")
                st.dataframe(result_df.head())

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv_bytes,
                    file_name="penguins_predictions.csv",
                    mime="text/csv",
                )
