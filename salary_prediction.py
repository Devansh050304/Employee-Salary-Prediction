import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("data/Salary_dataset.csv")

# Select features and target
X = data[["YearsExperience"]]
y = data["Salary"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
score = r2_score(y_test, predictions)

print("Model Accuracy (R² Score):", round(score, 2))

# Predict salary for 5 years of experience
experience = [[5]]
predicted_salary = model.predict(experience)

print(f"\nPredicted Salary for 5 Years Experience: ${predicted_salary[0]:,.2f}")

# Plot graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Employee Salary Prediction")
plt.legend()
plt.show()