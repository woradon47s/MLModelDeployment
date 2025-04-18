import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained model to model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Model trained and saved as model.pkl")
