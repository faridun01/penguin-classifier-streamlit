import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title('🎈 Hello ;)')
st.write("Let's do it")

# ----------- LOAD DATA -----------
df = pd.read_csv(
    'https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv'
)

# ----------- TRAIN MODEL -----------
# Target and features
target_col = "species"
feature_cols = ["island", "bill_length_mm", "bill_depth_mm",
                "flipper_length_mm", "body_mass_g", "sex"]

X = df[feature_cols]
y = df[target_col]

# One-hot encode categorical features (island, sex)
X_encoded = pd.get_dummies(X, drop_first=True)

# Train/test split (just to be a bit realistic, although we only use model)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Train a simple RandomForest
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Save columns of encoded X (needed to align user input)
model_columns = X_encoded.columns


# ----------- EXPANDER: DATA -----------
with st.expander("DATA"):
    st.write("OUR DATA")
    st.dataframe(df)

# ----------- EXPANDER: VISUALIZATION -----------
with st.expander("Visualization"):
    st.scatter_chart(
        data=df,
        x='bill_length_mm',
        y='body_mass_g',
        color='species'
    )

# ----------- SIDEBAR INPUTS -----------
st.sidebar.header("Input features")

island = st.sidebar.selectbox("Island", sorted(df["island"].unique()))

bill_length_mm = st.sidebar.slider(
    "Bill length (mm)",
    min_value=float(df.bill_length_mm.min()),
    max_value=float(df.bill_length_mm.max()),
    value=float(df.bill_length_mm.mean())
)

bill_depth_mm = st.sidebar.slider(
    "Bill depth (mm)",
    min_value=float(df.bill_depth_mm.min()),
    max_value=float(df.bill_depth_mm.max()),
    value=float(df.bill_depth_mm.mean())
)

flipper_length_mm = st.sidebar.slider(
    "Flipper length (mm)",
    min_value=float(df.flipper_length_mm.min()),
    max_value=float(df.flipper_length_mm.max()),
    value=float(df.flipper_length_mm.mean())
)

body_mass_g = st.sidebar.slider(
    "Body mass (g)",
    min_value=int(df.body_mass_g.min()),
    max_value=int(df.body_mass_g.max()),
    value=int(df.body_mass_g.mean())
)

sex = st.sidebar.selectbox("Gender", sorted(df["sex"].unique()))

user_input = {
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": sex
}

st.subheader("Your Input")
st.json(user_input)

# ----------- PREDICTION -----------
# Convert to DataFrame
user_df = pd.DataFrame([user_input])

# One-hot encode like training data
user_encoded = pd.get_dummies(user_df, drop_first=True)

# Align columns with training data (very important)
user_encoded = user_encoded.reindex(columns=model_columns, fill_value=0)

st.subheader("Prediction")

if st.button("Predict species"):
    pred = model.predict(user_encoded)[0]
    proba = model.predict_proba(user_encoded)[0]

    st.success(f"Predicted species: {pred}")

    proba_df = pd.DataFrame(
        [proba],
        columns=model.classes_
    )
    st.write("Class probabilities:")
    st.dataframe(proba_df)
