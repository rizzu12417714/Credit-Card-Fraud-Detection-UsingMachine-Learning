import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

print("\nLoading dataset...")

df = pd.read_csv(r"C:\Users\singh\Downloads\diabetes.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())

df = df.drop_duplicates()
df = df.dropna(subset=["Outcome"])
df = df[df["Outcome"].isin([0, 1])].copy()
df["Outcome"] = df["Outcome"].astype(int)

zero_not_allowed = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_not_allowed:
    df[col] = df[col].replace(0, np.nan)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset after cleaning:", df.shape)

print("\nDiabetes status:")
print("Non-diabetic patients:", (df["Outcome"] == 0).sum())
print("Diabetic patients:", (df["Outcome"] == 1).sum())

os.makedirs("project_outputs", exist_ok=True)

numeric_features = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

features = numeric_features

X = df[features]
y = df["Outcome"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=1
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = []
trained_models = {}
model_probabilities = {}
model_predictions = {}

print("\nTraining all five models...")

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    trained_models[name] = pipeline
    model_probabilities[name] = probabilities
    model_predictions[name] = predictions

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "project_outputs/model_comparison.csv",
    index=False
)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = model_predictions[best_name]
best_probabilities = model_probabilities[best_name]

print("\nBest model by test F1:", best_name)

print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Non-Diabetic", "Diabetic"],
    zero_division=0
))

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Accuracy Comparison of Five ML Models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("project_outputs/01_model_accuracy.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["F1 Score"])
plt.title("F1 Score Comparison of Five ML Models")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("project_outputs/02_model_f1_score.png")
plt.show()

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - " + best_name)
plt.colorbar()
plt.xticks([0, 1], ["Non-Diabetic", "Diabetic"])
plt.yticks([0, 1], ["Non-Diabetic", "Diabetic"])
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center"
        )

plt.tight_layout()
plt.savefig("project_outputs/03_confusion_matrix.png")
plt.show()

status_counts = df["Outcome"].value_counts().reindex(
    [0, 1], fill_value=0
)

plt.figure(figsize=(6, 5))
plt.bar(
    ["Non-Diabetic", "Diabetic"],
    status_counts.values
)
plt.title("Non-Diabetic vs Diabetic Patients")
plt.xlabel("Diabetes Status")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("project_outputs/04_diabetes_status.png")
plt.show()

plt.figure(figsize=(9, 6))

for name, probabilities in model_probabilities.items():

    fpr, tpr, thresholds = roc_curve(
        y_test, probabilities
    )

    auc = roc_auc_score(y_test, probabilities)

    plt.plot(
        fpr, tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves - Five Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("project_outputs/05_roc_curves.png")
plt.show()

plt.figure(figsize=(9, 6))

for name, probabilities in model_probabilities.items():

    precision_values, recall_values, thresholds = (
        precision_recall_curve(y_test, probabilities)
    )

    ap = average_precision_score(
        y_test, probabilities
    )

    plt.plot(
        recall_values,
        precision_values,
        label=f"{name} (AP={ap:.3f})"
    )

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("project_outputs/06_precision_recall.png")
plt.show()

rf_pipeline = trained_models["Random Forest"]
rf_model = rf_pipeline.named_steps["model"]

feature_names = (
    rf_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

clean_labels = [
    name.replace("numeric__", "")
    for name in importances.index
]

plt.figure(figsize=(9, 6))
plt.barh(
    clean_labels[::-1],
    importances.values[::-1]
)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("project_outputs/07_feature_importance.png")
plt.show()

df["glucose_group"] = pd.cut(
    df["Glucose"],
    bins=[0, 90, 110, 130, 150, 170, 250],
    labels=["<90", "90-110", "110-130",
            "130-150", "150-170", "170+"]
)

glucose_rate = (
    df.groupby("glucose_group", observed=False)["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(9, 5))
plt.bar(
    glucose_rate.index.astype(str),
    glucose_rate.values
)
plt.title("Diabetes Rate by Glucose Level")
plt.xlabel("Glucose Group")
plt.ylabel("Diabetic Patients (%)")
plt.tight_layout()
plt.savefig("project_outputs/08_rate_glucose.png")
plt.show()

df["age_group"] = pd.cut(
    df["Age"],
    bins=[20, 30, 40, 50, 60, 90],
    labels=["21-30", "31-40", "41-50",
            "51-60", "60+"],
    include_lowest=True
)

age_rate = (
    df.groupby("age_group", observed=False)["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(8, 5))
plt.plot(
    age_rate.index.astype(str),
    age_rate.values,
    marker="o"
)
plt.title("Diabetes Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Diabetic Patients (%)")
plt.tight_layout()
plt.savefig("project_outputs/09_rate_age.png")
plt.show()

correlation_columns = numeric_features + ["Outcome"]
correlation = df[correlation_columns].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
plt.imshow(
    correlation,
    aspect="auto",
    vmin=-1,
    vmax=1
)
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=60,
    ha="right"
)
plt.yticks(
    range(len(correlation.index)),
    correlation.index
)

for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        plt.text(
            j, i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center", va="center", fontsize=7
        )

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("project_outputs/10_correlation_heatmap.png")
plt.show()

df["bmi_group"] = pd.cut(
    df["BMI"],
    bins=[0, 18.5, 25, 30, 35, 40, 70],
    labels=["<18.5", "18.5-25", "25-30",
            "30-35", "35-40", "40+"]
)

bmi_rate = (
    df.groupby("bmi_group", observed=False)["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(8, 5))
plt.barh(
    bmi_rate.index.astype(str),
    bmi_rate.values
)
plt.title("Diabetes Rate by BMI Group")
plt.xlabel("Diabetic Patients (%)")
plt.ylabel("BMI Group")
plt.tight_layout()
plt.savefig("project_outputs/11_rate_bmi.png")
plt.show()

preg_rate = (
    df.groupby("Pregnancies")["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(8, 5))
plt.barh(
    preg_rate.index.astype(str),
    preg_rate.values
)
plt.title("Diabetes Rate by Number of Pregnancies")
plt.xlabel("Diabetic Patients (%)")
plt.ylabel("Pregnancies")
plt.tight_layout()
plt.savefig("project_outputs/12_rate_pregnancies.png")
plt.show()

df["bp_group"] = pd.cut(
    df["BloodPressure"],
    bins=[0, 60, 70, 80, 90, 130],
    labels=["<60", "60-70", "70-80",
            "80-90", "90+"]
)

bp_rate = (
    df.groupby("bp_group", observed=False)["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(9, 5))
plt.plot(
    bp_rate.index.astype(str),
    bp_rate.values,
    marker="o"
)
plt.title("Diabetes Rate by Blood Pressure")
plt.xlabel("Blood Pressure Group")
plt.ylabel("Diabetic Patients (%)")
plt.tight_layout()
plt.savefig("project_outputs/13_rate_blood_pressure.png")
plt.show()

df["insulin_group"] = pd.cut(
    df["Insulin"],
    bins=[0, 50, 100, 150, 200, 900],
    labels=["<50", "50-100", "100-150",
            "150-200", "200+"]
)

insulin_rate = (
    df.groupby("insulin_group", observed=False)["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(9, 5))
plt.plot(
    insulin_rate.index.astype(str),
    insulin_rate.values,
    marker="o"
)
plt.title("Diabetes Rate by Insulin Level")
plt.xlabel("Insulin Group")
plt.ylabel("Diabetic Patients (%)")
plt.tight_layout()
plt.savefig("project_outputs/14_rate_insulin.png")
plt.show()

non_diabetic_glucose = df.loc[
    df["Outcome"] == 0,
    "Glucose"
].dropna()

diabetic_glucose = df.loc[
    df["Outcome"] == 1,
    "Glucose"
].dropna()

plt.figure(figsize=(7, 5))
plt.boxplot(
    [non_diabetic_glucose, diabetic_glucose],
    showfliers=False
)
plt.xticks([1, 2], ["Non-Diabetic", "Diabetic"])
plt.title("Glucose Distribution by Diabetes Status")
plt.ylabel("Glucose")
plt.tight_layout()
plt.savefig("project_outputs/15_glucose_distribution.png")
plt.show()

non_diabetic_bmi = df.loc[
    df["Outcome"] == 0,
    "BMI"
].dropna()

diabetic_bmi = df.loc[
    df["Outcome"] == 1,
    "BMI"
].dropna()

plt.figure(figsize=(7, 5))
plt.boxplot(
    [non_diabetic_bmi, diabetic_bmi],
    showfliers=False
)
plt.xticks([1, 2], ["Non-Diabetic", "Diabetic"])
plt.title("BMI Distribution by Diabetes Status")
plt.ylabel("BMI")
plt.tight_layout()
plt.savefig("project_outputs/16_bmi_distribution.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.hist(
    df.loc[df["Outcome"] == 0, "Age"],
    bins=15,
    alpha=0.6,
    label="Non-Diabetic"
)
plt.hist(
    df.loc[df["Outcome"] == 1, "Age"],
    bins=15,
    alpha=0.6,
    label="Diabetic"
)
plt.title("Age Distribution by Diabetes Status")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.legend()
plt.tight_layout()
plt.savefig("project_outputs/17_age_histogram.png")
plt.show()

df["pedigree_group"] = pd.cut(
    df["DiabetesPedigreeFunction"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 3.0],
    labels=["0-0.2", "0.2-0.4", "0.4-0.6",
            "0.6-0.8", "0.8-1.0", "1.0+"],
    include_lowest=True
)

pedigree_rate = (
    df.groupby("pedigree_group", observed=False)["Outcome"]
    .mean() * 100
)

plt.figure(figsize=(8, 5))
plt.bar(
    pedigree_rate.index.astype(str),
    pedigree_rate.values
)
plt.title("Diabetes Rate by Pedigree Function")
plt.xlabel("Diabetes Pedigree Function Group")
plt.ylabel("Diabetic Patients (%)")
plt.tight_layout()
plt.savefig("project_outputs/18_rate_pedigree.png")
plt.show()

prediction_output = X_test.copy()
prediction_output["actual_status"] = y_test
prediction_output["predicted_status"] = best_predictions
prediction_output["diabetes_probability"] = best_probabilities

prediction_output.to_csv(
    "project_outputs/test_predictions.csv",
    index=False
)

print("\n========== NEW PATIENT PREDICTION ==========")
print("Enter the details of a new patient.")

try:
    new_patient = pd.DataFrame([{
        "Pregnancies": int(input("Number of pregnancies: ")),
        "Glucose": float(input("Glucose level: ")),
        "BloodPressure": float(input("Blood pressure: ")),
        "SkinThickness": float(input("Skin thickness: ")),
        "Insulin": float(input("Insulin level: ")),
        "BMI": float(input("BMI: ")),
        "DiabetesPedigreeFunction": float(
            input("Diabetes pedigree function: ")
        ),
        "Age": int(input("Age: "))
    }])

    result = best_model.predict(new_patient)[0]

    diabetes_probability = (
        best_model.predict_proba(new_patient)[0][1]
    )

    print("\n========== PREDICTION RESULT ==========")
    print(
        "Estimated diabetes probability:",
        round(diabetes_probability * 100, 2), "%"
    )

    if result == 1:
        print("Prediction: DIABETIC")
    else:
        print("Prediction: NON-DIABETIC")

except ValueError:
    print("Invalid input. Please enter valid numeric values.")

print("\nProject execution completed.")
print("All 18 graphs are saved in the project_outputs folder.")
print("Model comparison and test predictions are also saved.")
