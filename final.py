import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("\nLoading dataset...")

df = pd.read_csv(r"C:\Users\DELL\Desktop\int234PredictiveProject\creditcard.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.describe().T[["mean", "std", "min", "max"]].head(10))

df = df.drop_duplicates()
df = df.dropna()
df["Class"] = df["Class"].astype(int)

print("\nMissing values:", int(df.isnull().sum().sum()))
print("Dataset after cleaning:", df.shape)

print("\nTransaction status:")
print("Genuine transactions:", (df["Class"] == 0).sum())
print("Fraud transactions:", (df["Class"] == 1).sum())

os.makedirs("project_outputs", exist_ok=True)

fraud_df = df[df["Class"] == 1]
genuine_df = df[df["Class"] == 0].sample(5000, random_state=42)

data = pd.concat([fraud_df, genuine_df]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

print("\nSampled dataset used for the project:", data.shape)
print("Genuine in sample:", (data["Class"] == 0).sum())
print("Fraud in sample:", (data["Class"] == 1).sum())

v_features = ["V" + str(i) for i in range(1, 29)]
features = ["Time", "Amount"] + v_features

X = data[features]
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
}

results = []
trained_models = {}
model_probabilities = {}
model_predictions = {}

print("\nTraining all five models...")

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, probabilities)
    loss = log_loss(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Log Loss": loss
    })

    trained_models[name] = model
    model_probabilities[name] = probabilities
    model_predictions[name] = predictions

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))
    print("Log Loss:", round(loss, 4))

results_df = pd.DataFrame(results).sort_values(
    by="F1 Score",
    ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(4).to_string(index=False))

results_df.to_csv("project_outputs/model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = model_predictions[best_name]

print("\nBest model by F1 Score:", best_name)

print("\nClassification report:")
print(classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Genuine", "Fraud"],
    zero_division=0
))

plt.figure(figsize=(6, 5))
plt.bar(["Genuine", "Fraud"], data["Class"].value_counts().reindex([0, 1]).values)
plt.title("Genuine vs Fraud Transactions (Sample)")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("project_outputs/01_class_distribution.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Accuracy Comparison of Five ML Models")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("project_outputs/02_model_accuracy.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["F1 Score"])
plt.title("F1 Score Comparison of Five ML Models")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("project_outputs/03_model_f1_score.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Log Loss"])
plt.title("Log Loss Comparison of Five ML Models")
plt.xlabel("Model")
plt.ylabel("Log Loss")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("project_outputs/04_model_log_loss.png")
plt.show()

plt.figure(figsize=(9, 6))

for name, probabilities in model_probabilities.items():
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    auc = roc_auc_score(y_test, probabilities)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curves - Five Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("project_outputs/05_roc_curves.png")
plt.show()

cm = confusion_matrix(y_test, best_predictions, labels=[0, 1])

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - " + best_name)
plt.colorbar()
plt.xticks([0, 1], ["Genuine", "Fraud"])
plt.yticks([0, 1], ["Genuine", "Fraud"])
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig("project_outputs/06_confusion_matrix.png")
plt.show()

correlation = data[features + ["Class"]].corr()["Class"].drop("Class")
top_corr = correlation.reindex(
    correlation.abs().sort_values(ascending=False).head(12).index
)

plt.figure(figsize=(9, 6))
plt.barh(top_corr.index[::-1], top_corr.values[::-1])
plt.title("Top 12 Features Correlated with Fraud")
plt.xlabel("Correlation with Class")
plt.tight_layout()
plt.savefig("project_outputs/07_correlation_with_fraud.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.boxplot(
    [data.loc[data["Class"] == 0, "Amount"], data.loc[data["Class"] == 1, "Amount"]],
    showfliers=False
)
plt.xticks([1, 2], ["Genuine", "Fraud"])
plt.title("Transaction Amount by Type")
plt.ylabel("Amount")
plt.tight_layout()
plt.savefig("project_outputs/08_amount_distribution.png")
plt.show()

print("\n========== REGRESSION (PREDICTING TRANSACTION AMOUNT) ==========")

reg_features = ["Time"] + v_features
Xr = data[reg_features]
yr = data["Amount"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.20, random_state=42
)

amount_corr = data[reg_features + ["Amount"]].corr()["Amount"].drop("Amount")
top_feature = amount_corr.abs().idxmax()

print("Feature most correlated with Amount:", top_feature)
print("Correlation:", round(amount_corr[top_feature], 4))

regression_results = []

def evaluate_regression(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    regression_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

x1_train = Xr_train[[top_feature]].values
x1_test = Xr_test[[top_feature]].values

simple_model = LinearRegression()
simple_model.fit(x1_train, yr_train)
simple_pred = simple_model.predict(x1_test)
evaluate_regression("Simple Linear", yr_test, simple_pred)

poly = PolynomialFeatures(degree=2)
x1_train_poly = poly.fit_transform(x1_train)
x1_test_poly = poly.transform(x1_test)

poly_model = LinearRegression()
poly_model.fit(x1_train_poly, yr_train)
poly_pred = poly_model.predict(x1_test_poly)
evaluate_regression("Polynomial (degree 2)", yr_test, poly_pred)

multi_model = LinearRegression()
multi_model.fit(Xr_train, yr_train)
multi_pred = multi_model.predict(Xr_test)
evaluate_regression("Multiple Linear", yr_test, multi_pred)

regression_df = pd.DataFrame(regression_results)

print(regression_df.round(4).to_string(index=False))

regression_df.to_csv("project_outputs/regression_comparison.csv", index=False)

order = np.argsort(x1_test[:, 0])

plt.figure(figsize=(8, 5))
plt.scatter(x1_test[:, 0], yr_test, alpha=0.4, label="Actual")
plt.plot(x1_test[order, 0], simple_pred[order], color="red", label="Simple Linear")
plt.title("Simple Linear Regression: Amount vs " + top_feature)
plt.xlabel(top_feature)
plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.savefig("project_outputs/09_simple_linear_regression.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(x1_test[:, 0], yr_test, alpha=0.4, label="Actual")
plt.plot(x1_test[order, 0], poly_pred[order], color="green", label="Polynomial (degree 2)")
plt.title("Polynomial Regression: Amount vs " + top_feature)
plt.xlabel(top_feature)
plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.savefig("project_outputs/10_polynomial_regression.png")
plt.show()

plt.figure(figsize=(7, 6))
plt.scatter(yr_test, multi_pred, alpha=0.4)
limit = max(yr_test.max(), multi_pred.max())
plt.plot([0, limit], [0, limit], linestyle="--", color="red")
plt.title("Multiple Linear Regression: Actual vs Predicted Amount")
plt.xlabel("Actual Amount")
plt.ylabel("Predicted Amount")
plt.tight_layout()
plt.savefig("project_outputs/11_multiple_regression.png")
plt.show()

print("\n========== CLUSTERING (UNSUPERVISED LEARNING) ==========")

cluster_scaler = StandardScaler()
cluster_data = cluster_scaler.fit_transform(data[v_features])

inertia = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(cluster_data)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia, marker="o")
plt.title("K-Means Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.tight_layout()
plt.savefig("project_outputs/12_kmeans_elbow.png")
plt.show()

print("\nRandom initialization vs k-means++ (single run, 3 seeds):")
for seed in [1, 2, 3]:
    km_random = KMeans(n_clusters=3, init="random", n_init=1, random_state=seed).fit(cluster_data)
    km_plus = KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=seed).fit(cluster_data)
    print(
        "Seed", seed,
        "| random inertia:", round(km_random.inertia_, 1),
        "| k-means++ inertia:", round(km_plus.inertia_, 1)
    )

kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=42)
data["kmeans_cluster"] = kmeans.fit_predict(cluster_data)

print("\nK-Means clusters vs actual Class:")
print(pd.crosstab(data["kmeans_cluster"], data["Class"]))

plt.figure(figsize=(8, 6))
for cluster in sorted(data["kmeans_cluster"].unique()):
    subset = data[data["kmeans_cluster"] == cluster]
    plt.scatter(subset["V1"], subset["V2"], alpha=0.4, label="Cluster " + str(cluster))
plt.title("K-Means Clusters (K=2) on V1 and V2")
plt.xlabel("V1")
plt.ylabel("V2")
plt.legend()
plt.tight_layout()
plt.savefig("project_outputs/13_kmeans_clusters.png")
plt.show()

agglo = AgglomerativeClustering(n_clusters=2, linkage="average")
data["hierarchical_cluster"] = agglo.fit_predict(cluster_data)

print("\nHierarchical (average linkage) clusters vs actual Class:")
print(pd.crosstab(data["hierarchical_cluster"], data["Class"]))

dendro_sample = pd.concat([
    data[data["Class"] == 0].sample(100, random_state=1),
    data[data["Class"] == 1].sample(50, random_state=1)
])

dendro_data = StandardScaler().fit_transform(dendro_sample[v_features])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, method in zip(axes.flatten(), ["single", "complete", "average", "centroid"]):
    Z = linkage(dendro_data, method=method)
    dendrogram(Z, ax=ax, no_labels=True)
    ax.set_title(method.capitalize() + " Linkage")
    ax.set_ylabel("Distance")

plt.suptitle("Hierarchical Clustering Dendrograms")
plt.tight_layout()
plt.savefig("project_outputs/14_dendrograms.png")
plt.show()

prediction_output = X_test.copy()
prediction_output["actual_status"] = y_test
prediction_output["predicted_status"] = best_predictions
prediction_output["fraud_probability"] = model_probabilities[best_name]

prediction_output.to_csv("project_outputs/test_predictions.csv", index=False)

print("\n========== SAMPLE TRANSACTION PREDICTIONS ==========")

sample = X_test.sample(5, random_state=1)
sample_scaled = scaler.transform(sample)
sample_probabilities = best_model.predict_proba(sample_scaled)[:, 1]
sample_results = best_model.predict(sample_scaled)

for i in range(len(sample)):
    print(
        "Transaction", i + 1,
        "| Amount:", round(sample["Amount"].iloc[i], 2),
        "| Fraud probability:", round(sample_probabilities[i] * 100, 2), "%",
        "| Prediction:", "FRAUD" if sample_results[i] == 1 else "GENUINE",
        "| Actual:", "FRAUD" if y_test.loc[sample.index[i]] == 1 else "GENUINE"
    )

print("\nProject execution completed.")
print("All 14 graphs, model comparison, regression results and predictions are saved in the project_outputs folder.")