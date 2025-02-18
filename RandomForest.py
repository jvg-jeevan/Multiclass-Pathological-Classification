import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def process(path):
    # Load the dataset
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Save results
    results_df = pd.DataFrame({'ID': range(1, len(y_pred) + 1), 'Predicted Value': y_pred})
    results_df.to_csv("results/resultRandomForest.csv", index=False)

    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(y_test, y_pred)

    # Print metrics
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2: {r2}")
    print(f"RMSE: {rmse}")
    print(f"Accuracy: {accuracy}")

    # Save metrics
    metrics_df = pd.DataFrame({
        'Parameter': ['MSE', 'MAE', 'R2', 'RMSE', 'Accuracy'],
        'Value': [mse, mae, r2, rmse, accuracy]
    })
    metrics_df.to_csv("results/RandomForestMetrics.csv", index=False)

    # Plot metrics
    plt.figure()
    metrics_df.plot(kind='bar', x='Parameter', y='Value', legend=False)
    plt.xlabel('Parameter')
    plt.ylabel('Value')
    plt.title('Random Forest Metrics')
    plt.savefig('results/RandomForestMetricsValue.png')
    plt.show()
