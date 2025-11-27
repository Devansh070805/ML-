import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score, mean_squared_error,accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.datasets import fetch_california_housing
import statsmodels.api as sm
from scipy.special import expit  


np.random.seed(42)
n_samples = 1000
x1 = np.random.randn(n_samples)
x2 = x1 + np.random.normal(0, 0.1, n_samples)
x3 = x1 + x2 + np.random.normal(0, 0.1, n_samples)
x4 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)
x5 = x2 + x3 + np.random.normal(0, 0.1, n_samples)
x6 = 2 * x1 - x3 + np.random.normal(0, 0.1, n_samples)
x7 = x4 + x5 + x6 + np.random.normal(0, 0.1, n_samples)
X = np.column_stack([x1, x2, x3, x4, x5, x6, x7])
y = 3*x1 + 2*x2 - 1.5*x3 + 0.5*x4 + np.random.normal(0, 0.5, n_samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def ridge_cost(X, y, weights, lam):
    n = len(y)
    y_pred = X.dot(weights)
    cost = (1/(2*n)) * np.sum((y_pred - y)**2) + (lam/(2*n)) * np.sum(weights[1:]**2)
    return cost

def ridge_gradient_descent(X, y, lr, lam, epochs=1000):
    n, m = X.shape
    weights = np.zeros(m)
    cost_history = []
    for _ in range(epochs):
        y_pred = X.dot(weights)
        gradient = (1/n) * X.T.dot(y_pred - y) + (lam/n) * np.r_[0, weights[1:]]
        weights -= lr * gradient
        cost_history.append(ridge_cost(X, y, weights, lam))
        if np.isnan(cost_history[-1]) or np.isinf(cost_history[-1]):
            return None, None  # Diverged
    return weights, cost_history

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
lambdas = [1e-15, 1e-10, 1e-5, 1e-3, 0, 1, 10, 20]
best_r2 = -np.inf
best_cost = np.inf
best_params = None
best_weights = None
best_history = None

for lr in learning_rates:
    for lam in lambdas:
        weights, cost_history = ridge_gradient_descent(X_train_b, y_train, lr, lam, epochs=1000)
        if weights is None:  
            continue
        y_pred = X_test_b.dot(weights)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            continue
        cost = ridge_cost(X_test_b, y_test, weights, lam)
        r2 = r2_score(y_test, y_pred)
        if (r2 > best_r2) or (r2 == best_r2 and cost < best_cost):
            best_r2 = r2
            best_cost = cost
            best_params = (lr, lam)
            best_weights = weights
            best_history = cost_history
print(f"Best Learning Rate: {best_params[0]}")
print(f"Best Regularization (λ): {best_params[1]}")
print(f"Minimum Cost: {best_cost:.6f}")
print(f"Maximum R² Score: {best_r2:.6f}")

y_pred_train = X_train_b.dot(best_weights)
y_pred_test = X_test_b.dot(best_weights)
print(f"Training R² Score: {r2_score(y_train, y_pred_train):.6f}")
print(f"Testing  R² Score: {r2_score(y_test, y_pred_test):.6f}")
plt.figure(figsize=(7,5))
plt.plot(best_history)
plt.title(f"Cost over Iterations (LR={best_params[0]}, λ={best_params[1]})")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
print("\nSample Predictions (Actual vs Predicted):")
for actual, pred in zip(y_test[:10], X_test_b.dot(best_weights)[:10]):
    print(f"Actual: {actual:.3f}   Predicted: {pred:.3f}")

df = sm.datasets.get_rdataset("Hitters", "ISLR").data
df = df.dropna()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.Categorical(df[col]).codes

y = df['Salary']
X = df.drop(columns=['Salary'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr = LinearRegression().fit(X_train_scaled, y_train)
ridge = Ridge(alpha=0.5748).fit(X_train_scaled, y_train)
lasso = Lasso(alpha=0.5748, max_iter=10000).fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)

def eval_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<20} →  MSE: {mse:.3f}   |   R²: {r2:.3f}")
    return r2

r2_lr = eval_model("Linear Regression", y_test, y_pred_lr)
r2_ridge = eval_model("Ridge Regression", y_test, y_pred_ridge)
r2_lasso = eval_model("Lasso Regression", y_test, y_pred_lasso)

best_model = max([(r2_lr, "Linear Regression"), (r2_ridge, "Ridge Regression"), (r2_lasso, "Lasso Regression")])[1]
print(f"\nBest Performing Model: {best_model}")

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge_alphas = np.logspace(-1, 2, 20)
ridge_cv = RidgeCV(alphas=ridge_alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
ridge_pred = ridge_cv.predict(X_test_scaled)

lasso_alphas = np.logspace(-1, 1, 20)
lasso_cv = LassoCV(alphas=lasso_alphas, cv=5, max_iter=5000, tol=0.01, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
lasso_pred = lasso_cv.predict(X_test_scaled)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

print("Ridge Regression:")
print(f"Best Alpha: {ridge_cv.alpha_}")
print(f"R² Score: {ridge_r2:.4f}")
print(f"RMSE: {ridge_rmse:.4f}\n")

print("Lasso Regression:")
print(f"Best Alpha: {lasso_cv.alpha_}")
print(f"R² Score: {lasso_r2:.4f}")
print(f"RMSE: {lasso_rmse:.4f}\n")

if ridge_r2 > lasso_r2:
    print("Ridge Regression performs better (handles correlated features well).")
else:
    print("Lasso Regression performs better (good for feature selection).")

iris = load_iris()
X = iris.data
y = iris.target
class_labels = iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]


def sigmoid(z):
    return expit(z)


def train_ovr_logistic_regression(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    weights = np.zeros((len(classes), n_features))
    
    for i, c in enumerate(classes):
        y_binary = np.where(y == c, 1, 0)
        w = np.zeros(n_features)
        for _ in range(epochs):
            z = X.dot(w)
            y_pred = sigmoid(z)
            gradient = (1/n_samples) * X.T.dot(y_pred - y_binary)
            w -= lr * gradient
        weights[i] = w
    return weights


def predict_ovr(X, weights):
    probs = sigmoid(X.dot(weights.T))
    return np.argmax(probs, axis=1)


weights = train_ovr_logistic_regression(X_train_b, y_train, lr=0.1, epochs=2000)


y_pred = predict_ovr(X_test_b, weights)


acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))


print("\nLearned weights (per class):")
for i, c in enumerate(class_labels):
    print(f"{c}: {weights[i]}")