from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def run_advanced_knn():
    # 📥 1. Load and prepare the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # 🔀 2. Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # ⚖️ 3. Scale features using StandardScaler[cite: 1]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 👤 4. User Interaction: Let the user choose a custom K value
    print("--- Interactive KNN Model ---")
    try:
        user_k = int(input("Enter a K value for the classifier (e.g., 1, 3, 5, 15): "))
    except ValueError:
        user_k = 5
        print("Invalid input. Defaulting to K = 5.")

    # ⚙️ 5. Train and evaluate with the user-selected K
    model = KNeighborsClassifier(n_neighbors=user_k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    print(f"\n📊 Results for K = {user_k}:")
    acc = accuracy_score(y_test, predictions) * 100
    print(f"Accuracy: {acc:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    # 🚀 6. Enhancement: Automated Optimal K Search (The Elbow Concept)
    print("\n--- Automated Optimization Scan (K from 1 to 15) ---")
    best_k = 1
    best_acc = 0
    for k in range(1, 16):
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train_scaled, y_train)
        temp_preds = temp_model.predict(X_test_scaled)
        temp_acc = accuracy_score(y_test, temp_preds)
        if temp_acc > best_acc:
            best_acc = temp_acc
            best_k = k

    print(f"✨ Optimal K found during scan: {best_k} with an accuracy of {best_acc * 100:.2f}%")


if __name__ == "__main__":
    run_advanced_knn()