import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA

file_path = "USA_Housing.csv"
data = pd.read_csv(file_path)

X = data.drop("Price", axis=1).values
y = data["Price"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = []
betas = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

    beta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ (X_train_bias.T @ y_train)

    y_pred = X_test_bias @ beta

    score = r2_score(y_test, y_pred)
    r2_scores.append(score)
    betas.append(beta)

for i, score in enumerate(r2_scores, start=1):
    print(f"Fold {i}: R2 Score = {score:.4f}")

best_index = np.argmax(r2_scores)
best_beta = betas[best_index]
print("\nBest Fold:", best_index + 1, " | R2 Score:", r2_scores[best_index])

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
X_test_bias = np.c_[np.ones(X_test.shape[0]), X_test]

beta_final = np.linalg.inv(X_train_bias.T @ X_train_bias) @ (X_train_bias.T @ y_train)

y_pred_final = X_test_bias @ beta_final

final_r2 = r2_score(y_test, y_pred_final)
print("\nFinal R² Score on 70/30 split:", final_r2)

data = pd.read_csv("USA_Housing.csv")

X = data.drop("Price", axis=1).values
y = data["Price"].values.reshape(-1,1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

X_train = np.c_[np.ones((X_train.shape[0],1)), X_train]
X_val = np.c_[np.ones((X_val.shape[0],1)), X_val]
X_test = np.c_[np.ones((X_test.shape[0],1)), X_test]

alphas = [0.001, 0.01, 0.1, 1]
betas = []
val_scores = []
test_scores = []

for lr in alphas:
    beta = np.zeros((X_train.shape[1],1))
    for _ in range(1000):
        grad = X_train.T @ (X_train @ beta - y_train) / len(y_train)
        beta -= lr * grad
    betas.append(beta)
    val_pred = X_val @ beta
    test_pred = X_test @ beta
    val_scores.append(r2_score(y_val, val_pred))
    test_scores.append(r2_score(y_test, test_pred))

for i, lr in enumerate(alphas):
    print(f"LR={lr}: Val R2={val_scores[i]:.4f}, Test R2={test_scores[i]:.4f}")

best_idx = np.argmax(val_scores)
print("\nBest LR:", alphas[best_idx])
print("Best Validation R2:", val_scores[best_idx])
print("Test R2 with Best LR:", test_scores[best_idx])
print("Best Beta:\n", betas[best_idx].flatten())

cols = ["symboling", "normalized_losses","make","fuel_type","aspiration","num_doors","body_style","drive_wheels",
        "engine_location","wheel_base","length","width","height","curb_weight","engine_type","num_cylinders",
        "engine_size","fuel_system","bore","stroke","compression_ratio","horsepower","peak_rpm","city_mpg",
        "highway_mpg","price"]

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",
                   names=cols)

data.replace("?", np.nan, inplace=True)

for c in data.columns:
    if c != "price":
        if data[c].dtype == "object":
            data[c].fillna(data[c].mode()[0], inplace=True)
        else:
            data[c] = pd.to_numeric(data[c], errors="coerce")
            data[c].fillna(data[c].mean(), inplace=True)

data = data.dropna(subset=["price"])
data["price"] = pd.to_numeric(data["price"], errors="coerce")

map_doors = {"two":2, "four":4}
map_cyl = {"two":2,"three":3,"four":4,"five":5,"six":6,"eight":8,"twelve":12}
data["num_doors"] = data["num_doors"].replace(map_doors)
data["num_cylinders"] = data["num_cylinders"].replace(map_cyl)

data = pd.get_dummies(data, columns=["body_style","drive_wheels"])

for c in ["make","aspiration","engine_location","fuel_type"]:
    le = LabelEncoder()
    data[c] = le.fit_transform(data[c])

data["fuel_system"] = data["fuel_system"].apply(lambda x: 1 if "pfi" in str(x) else 0)
data["engine_type"] = data["engine_type"].apply(lambda x: 1 if "ohc" in str(x) else 0)

X = data.drop("price", axis=1).values
y = data["price"].values

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("R2 without PCA:", r2_score(y_test, y_pred))

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

reg2 = LinearRegression()
reg2.fit(X_train_r, y_train_r)
y_pred_r = reg2.predict(X_test_r)
print("R2 with PCA:", r2_score(y_test_r, y_pred_r))