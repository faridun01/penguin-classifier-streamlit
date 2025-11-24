import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Penguins Classifier", page_icon="🐧", layout="centered")

st.title("🐧 Penguins classifier")
st.write("Predict penguin species using 4 features:")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv"
    )
    return df

df = load_data()

# ----------------- TRAIN MODELS (KNN, DT, ENSEMBLE) -----------------
@st.cache_resource
def train_models(df):
    target_col = "species"
    feature_cols = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)

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

    return trained_models, metrics_df, best_model_name


models, metrics_df, best_model_name = train_models(df)

# ----------------- SHOW MODEL ACCURACY -----------------
with st.expander("Model performance"):
    st.write("Accuracy of each model (using only 4 numeric features):")
    st.dataframe(metrics_df.style.format({"Accuracy": "{:.3f}"}))
    st.success(
        f"Best model: {best_model_name} "
        f"(Accuracy = {metrics_df.iloc[0]['Accuracy']:.3f})"
    )

# ----------------- SIDEBAR INPUT (4 FEATURES ONLY) -----------------
st.sidebar.header("Input features")

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

# collect input
user_input = {
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
}

st.subheader("Your input")
st.json(user_input)

# ----------------- CHOOSE MODEL -----------------
selected_model_name = st.selectbox(
    "Choose model for prediction",
    options=list(models.keys()),
    index=list(models.keys()).index(best_model_name),
)
selected_model = models[selected_model_name]

# ----------------- PREDICTION BUTTON -----------------
st.subheader("Prediction")

if st.button("Predict penguin species"):
    user_df = pd.DataFrame([user_input])  # 1 row, 4 columns
    pred = selected_model.predict(user_df)[0]

    # not all classifiers guarantee predict_proba, но у KNN, DT, Voting в sklearn оно есть
    proba = selected_model.predict_proba(user_df)[0]

    st.success(f"Model: {selected_model_name}")
    st.success(f"Predicted species: {pred}")

    proba_df = pd.DataFrame([proba], columns=selected_model.classes_)
    st.write("Class probabilities:")
    st.dataframe(proba_df)
