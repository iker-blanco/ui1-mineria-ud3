import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# 1

# 1.1
# Load the training and test data
train_data = pd.read_csv("data/vino_entrenamiento.csv")
test_data = pd.read_csv("data/vino_prueba.csv")

# 1.2
# Separate features and labels
X_train = train_data.drop(
    "class", axis=1
)  # Class is the target column, so we drop it from the features
y_train = train_data["class"]
X_test = test_data.drop("class", axis=1)
y_test = test_data["class"]

# Initialize the Gaussian Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train, y_train)

# Predict the labels
y_pred = nb_model.predict(X_test)


# List the real labels and the predicted values
print("Etiquetas reales:\n", y_test.values)
print("Valores predichos:\n", y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Porcentaje de acierto: {accuracy * 100:.2f}%")
accuracy_without_pca = accuracy


# Calculate the classification errors by comparing the real labels with the predicted values
# and counting the number of mismatches
failures = (y_test != y_pred).sum()
print("Fallos de clasificación:", failures)
failures_without_pca = failures

# Another way to calculate the classification errors is by using the confusion matrix
# which shows the number of true positives, true negatives, false positives, and false negatives
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusión:\n", conf_matrix)


# Represent the decision boundary using only the features "Proline" and "OD280/OD315 of diluted wines"
columns = ["Proline", "OD280/OD315 of diluted wines"]
X_train = train_data[columns]
y_train = train_data["class"]
X_test = test_data[columns]
y_test = test_data["class"]

# Encode the target labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train_encoded)
# Create a meshgrid of points to make predictions
x_min, x_max = X_test["Proline"].min() - 50, X_test["Proline"].max() + 50
y_min, y_max = (
    X_test["OD280/OD315 of diluted wines"].min() - 0.5,
    X_test["OD280/OD315 of diluted wines"].max() + 0.5,
)
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Make predictions on the meshgrid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

# Plot the training points
scatter = plt.scatter(
    X_test["Proline"],
    X_test["OD280/OD315 of diluted wines"],
    c=y_test_encoded,
    cmap="coolwarm",
    edgecolor="k",
    s=20,
)

# Highlight the misclassified points
y_pred = model.predict(X_test)
errors = y_test_encoded != y_pred
plt.scatter(
    X_test.iloc[errors, 0],
    X_test.iloc[errors, 1],
    c="red",
    s=50,
    label="Errores de Clasificación",
    edgecolors="black",
    marker="o",
)

# Format the plot to make it more readable :)
plt.title("Frontera de decisión Naïve-Bayes")
plt.xlabel("Proline")
plt.ylabel("OD280/OD315 of diluted wines")
plt.colorbar(scatter, ticks=[0, 1, 2], label="Clases (Joven, Crianza, Reserva)")
plt.legend()

# Show the plot
plt.show()


# 2


# 2.1
# Load the training data
data = pd.read_csv("data/vino_entrenamiento.csv")

# Separate the features and the target variable
X = data.drop(
    columns=["class"]
)  # Class is the target variable, so we drop it from the features
y = data["class"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca.fit(X_scaled)

# Variance explained by each component
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Plot the explained variance
plt.figure(figsize=(10, 5))
plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance,
    alpha=0.5,
    align="center",
    label="Varianza individual explicada",
)
plt.step(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    where="mid",
    label="Varianza acumulada explicada",
)
# Add labels and title
plt.ylabel("Proporción de Varianza Explicada")
plt.xlabel("Componentes Principales")
plt.legend(loc="best")
plt.tight_layout()
# Show the plot
plt.show()

# Determine the number of principal components to keep at least 80% of the variance
num_components = (cumulative_variance >= 0.8).argmax() + 1
print(
    f"Número de componentes principales para mantener al menos 80% de la varianza: {num_components}"
)
# In our case, we need to keep 5 principal components to retain at least 80% of the variance

# 2.2

# Load the training and test data
X_train = train_data.drop(columns=["class"])
y_train = train_data["class"]
X_test = test_data.drop(columns=["class"])
y_test = test_data["class"]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Number of components chosen from the previous analysis
# Run PCA with the chosen number of components
pca = PCA(n_components=num_components)
pca.fit(X_train_scaled)

X_train_pca = pca.transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train the Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_pca, y_train)

# Predict the labels
y_pred = nb_model.predict(X_test_pca)


# 2.3

# List the real labels and the predicted values
print("Etiquetas reales:\n", y_test.values)
print("Valores predichos:\n", y_pred)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Porcentaje de acierto: {accuracy * 100:.2f}%")

# Calculate the classification errors
conf_matrix = confusion_matrix(y_test, y_pred)
failures = sum(y_test != y_pred)
print(f"Fallos de clasificación: {failures}")
print("Matriz de Confusión:\n", conf_matrix)


# Print the results with and without PCA
print("Resultados con PCA:")
print(f"Porcentaje de acierto con PCA: {accuracy * 100:.2f}%")
print(f"Fallos de clasificación con PCA: {failures}")

print("Resultados sin PCA:")
print(f"Porcentaje de acierto sin PCA: {accuracy_without_pca * 100:.2f}%")
print(f"Fallos de clasificación sin PCA: {failures_without_pca}")

# 3

# Choose the features for the scatter plot
total_phenols = data["Total phenols"]
flavanoids = data["Flavanoids"]

# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(total_phenols, flavanoids, color="blue", alpha=0.5)
plt.title("Relación entre Total Phenols y Flavanoids")
plt.xlabel("Total Phenols")
plt.ylabel("Flavanoids")
plt.grid(True)
plt.show()


# 3.2

X = data["Total phenols"].values.reshape(-1, 1)
Y = data["Flavanoids"].values

# Crete a list of alphas to try with Kernel Ridge Regression
alphas = [0.04, 0.2, 0.8]
models = [KernelRidge(alpha=alpha, kernel="rbf") for alpha in alphas]

# Fit the models and predict
predictions = [model.fit(X, Y).predict(X) for model in models]

# Calculate the MSE for each alpha
mse_scores = [mean_squared_error(Y, pred) for pred in predictions]

# Print the MSE for each alpha and determine the best alpha
for alpha, mse in zip(alphas, mse_scores):
    print(f"MSE for alpha={alpha}: {mse:.4f}")
best_alpha = alphas[np.argmin(mse_scores)]
print(f"Best alpha: {best_alpha}")

# Plot the results
x_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
predictions_plot = [model.predict(x_plot) for model in models]

for pred, alpha in zip(predictions_plot, alphas):
    plt.plot(x_plot, pred, label=f"Alpha={alpha}")

plt.scatter(X, Y, color="red", label="Data")
plt.title("Kernel Ridge Regression with Gaussian Kernel")
plt.xlabel("Total Phenols")
plt.ylabel("Flavanoids")
plt.legend()
plt.show()


# 3.3

# Create a pipeline for polynomial regression of degree 5
degree = 5
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Use the best alpha determined in the previous step to calculate the best gamma
best_alpha = alphas[np.argmin(mse_scores)]
best_gamma = 1 / (2 * best_alpha**2)  # Calculate gamma from alpha

# Create a Kernel Ridge model with the best gamma
kernel_model = KernelRidge(alpha=1e-6, kernel="rbf", gamma=best_gamma)

# Fit the models and predict
poly_model.fit(X, Y)
y_poly = poly_model.predict(x_plot)

kernel_model.fit(X, Y)
y_kernel = kernel_model.predict(x_plot)

# Calculate MSE for both models
mse_poly = mean_squared_error(Y, poly_model.predict(X))
mse_kernel = mean_squared_error(Y, kernel_model.predict(X))

print(f"MSE for Polynomial Regression: {mse_poly:.4f}")
print(f"MSE for Kernel Ridge Regression: {mse_kernel:.4f}")


# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color="darkorange", label="Datos reales")
plt.plot(
    x_plot, y_kernel, label=f"Kernel Ridge con gamma = {best_gamma:.2f}", linewidth=2
)
plt.plot(
    x_plot,
    y_poly,
    label=f"Polinomial grado 5 (MSE={mse_poly:.2f})",
    linestyle="--",
    linewidth=2,
    color="black",
)
plt.xlabel("Total Phenols")
plt.ylabel("Flavanoids")
plt.title("Comparación de Modelos de Regresión")
plt.legend()
plt.show()
