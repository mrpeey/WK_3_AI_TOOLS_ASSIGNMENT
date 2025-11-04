"""
Classical Machine Learning with Scikit-learn
Iris Species Classification Using Decision Tree

Dataset: Iris Species Dataset

Goals:
1. Preprocess the data (handle missing values, encode labels)
2. Train a decision tree classifier to predict iris species
3. Evaluate using accuracy, precision, and recall

Author: ML Project
Date: November 3, 2025
"""

# ============================================================================
# 1. Import Required Libraries
# ============================================================================
print("="*70)
print("CLASSICAL MACHINE LEARNING - IRIS SPECIES CLASSIFICATION")
print("="*70)
print("\n[Step 1] Importing required libraries...\n")

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Dataset loading
from sklearn.datasets import load_iris

# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Machine learning model
from sklearn.tree import DecisionTreeClassifier

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("✓ All libraries imported successfully!")


# ============================================================================
# 2. Load the Iris Dataset
# ============================================================================
print("\n" + "="*70)
print("[Step 2] Loading the Iris dataset...")
print("="*70 + "\n")

# Load the Iris dataset from scikit-learn
iris = load_iris()

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add the target column (species)
df['species'] = iris.target

# Display basic information about the dataset
print("✓ Dataset loaded successfully!")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")  # Excluding target column
print(f"\nFeature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")


# ============================================================================
# 3. Explore the Dataset
# ============================================================================
print("\n" + "="*70)
print("[Step 3] Exploring the dataset...")
print("="*70 + "\n")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\n" + "-"*70 + "\n")

# Display statistical summary of the features
print("Statistical summary of features:")
print(df.describe())

print("\n" + "-"*70 + "\n")

# Check data types
print("Data types:")
print(df.dtypes)

print("\n" + "-"*70 + "\n")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())

print("\n" + "-"*70 + "\n")

# Check the distribution of target classes
print("Distribution of species:")
print(df['species'].value_counts().sort_index())
print("\nMapping: 0 = setosa, 1 = versicolor, 2 = virginica")


# ============================================================================
# 4. Preprocess the Data
# ============================================================================
print("\n" + "="*70)
print("[Step 4] Preprocessing the data...")
print("="*70 + "\n")

# Step 1: Handle missing values
# Check if there are any missing values
missing_count = df.isnull().sum().sum()

if missing_count > 0:
    print(f"Found {missing_count} missing values.")
    # For numerical features, we could fill with median or mean
    # For this dataset, we'll drop rows with missing values
    df_clean = df.dropna()
    print(f"Dropped rows with missing values. New shape: {df_clean.shape}")
else:
    print("✓ No missing values found. Dataset is clean!")
    df_clean = df.copy()

print("\n" + "-"*70 + "\n")

# Step 2: Separate features (X) and target (y)
# Features: all columns except 'species'
X = df_clean.drop('species', axis=1)

# Target: the 'species' column (already encoded as 0, 1, 2)
y = df_clean['species']

print("✓ Features (X) shape:", X.shape)
print("✓ Target (y) shape:", y.shape)

print("\n" + "-"*70 + "\n")

# Step 3: Verify encoding
# The Iris dataset from sklearn already has encoded labels (0, 1, 2)
# If we had string labels, we would use LabelEncoder:
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

print("Target variable is already encoded:")
print(f"Unique values: {sorted(y.unique())}")
print(f"\nClass distribution:")
print(y.value_counts().sort_index())

print("\n✓ Data preprocessing completed successfully!")


# ============================================================================
# 5. Split the Data into Training and Testing Sets
# ============================================================================
print("\n" + "="*70)
print("[Step 5] Splitting data into training and testing sets...")
print("="*70 + "\n")

# Split the data: 80% training, 20% testing
# random_state=42 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution in both sets
)

print("✓ Data split completed!")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print("\n" + "-"*70 + "\n")

# Verify class distribution in train and test sets
print("Class distribution in training set:")
print(y_train.value_counts().sort_index())

print("\nClass distribution in testing set:")
print(y_test.value_counts().sort_index())


# ============================================================================
# 6. Train a Decision Tree Classifier
# ============================================================================
print("\n" + "="*70)
print("[Step 6] Training the Decision Tree Classifier...")
print("="*70 + "\n")

# Initialize the Decision Tree Classifier
# We'll use default parameters first, but you can tune these for better performance
dt_classifier = DecisionTreeClassifier(
    criterion='gini',      # Measure of split quality ('gini' or 'entropy')
    max_depth=None,        # Maximum depth of the tree (None = unlimited)
    min_samples_split=2,   # Minimum samples required to split a node
    min_samples_leaf=1,    # Minimum samples required at a leaf node
    random_state=42        # For reproducibility
)

print("Decision Tree Classifier initialized with parameters:")
for key, value in dt_classifier.get_params().items():
    print(f"  {key}: {value}")

print("\n" + "-"*70 + "\n")

# Train the model on the training data
print("Training the model...")
dt_classifier.fit(X_train, y_train)
print("✓ Model training completed!")

print("\n" + "-"*70 + "\n")

# Display tree information
print("Tree Structure Information:")
print(f"  Tree depth: {dt_classifier.get_depth()}")
print(f"  Number of leaves: {dt_classifier.get_n_leaves()}")
print(f"  Number of features used: {dt_classifier.n_features_in_}")


# ============================================================================
# 7. Make Predictions
# ============================================================================
print("\n" + "="*70)
print("[Step 7] Making predictions on the test set...")
print("="*70 + "\n")

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

print("✓ Predictions made on test set!")
print(f"Number of predictions: {len(y_pred)}")

print("\n" + "-"*70 + "\n")

# Display first 10 predictions vs actual values
print("First 10 predictions vs actual values:")
comparison_df = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': y_pred[:10],
    'Match': ['✓' if a == p else '✗' for a, p in zip(y_test.values[:10], y_pred[:10])]
})
print(comparison_df.to_string(index=False))

print("\n" + "-"*70 + "\n")

# Also get prediction probabilities for each class
y_pred_proba = dt_classifier.predict_proba(X_test)
print("Prediction probabilities for first 5 samples:")
print("(Columns represent: setosa, versicolor, virginica)")
proba_df = pd.DataFrame(
    y_pred_proba[:5],
    columns=iris.target_names
)
print(proba_df)


# ============================================================================
# 8. Evaluate the Model
# ============================================================================
print("\n" + "="*70)
print("[Step 8] Evaluating the model...")
print("="*70 + "\n")

# Calculate evaluation metrics

# 1. Accuracy: Overall correctness of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("→ Interpretation: Percentage of correct predictions out of all predictions")

print("\n" + "-"*70 + "\n")

# 2. Precision: How many predicted positives are actually positive
# Using 'weighted' average to account for class imbalance
precision = precision_score(y_test, y_pred, average='weighted')
print(f"PRECISION (weighted): {precision:.4f}")
print("→ Interpretation: Of all predicted species, how many were correctly identified")

print("\n" + "-"*70 + "\n")

# 3. Recall: How many actual positives were correctly predicted
# Using 'weighted' average to account for class imbalance
recall = recall_score(y_test, y_pred, average='weighted')
print(f"RECALL (weighted): {recall:.4f}")
print("→ Interpretation: Of all actual species, how many were correctly identified")

print("\n" + "-"*70 + "\n")

# 4. F1-Score: Harmonic mean of precision and recall
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1-SCORE (weighted): {f1:.4f}")
print("→ Interpretation: Balance between precision and recall")

print("\n" + "-"*70 + "\n")

# 5. Per-class metrics using classification report
print("DETAILED CLASSIFICATION REPORT:")
print("="*70)
print(classification_report(
    y_test, 
    y_pred,
    target_names=iris.target_names,
    digits=4
))
print("Note: Support shows the number of actual occurrences of each class")

print("\n" + "-"*70 + "\n")

# 6. Confusion Matrix
# Shows the number of correct and incorrect predictions for each class
cm = confusion_matrix(y_test, y_pred)

print("CONFUSION MATRIX:")
print("="*70)
print(cm)

print("\n")

# Create a more readable confusion matrix using pandas
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {name}" for name in iris.target_names],
    columns=[f"Predicted {name}" for name in iris.target_names]
)
print("Confusion Matrix (detailed):")
print(cm_df)

print("\n→ Interpretation:")
print("  - Diagonal elements: Correctly classified samples")
print("  - Off-diagonal elements: Misclassified samples")


# ============================================================================
# 9. Visualize the Confusion Matrix
# ============================================================================
print("\n" + "="*70)
print("[Step 9] Creating visualizations...")
print("="*70 + "\n")

# Visualize the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_df,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=True,
    square=True,
    linewidths=1,
    linecolor='black'
)
plt.title('Confusion Matrix - Decision Tree Classifier\nIris Species Classification', 
          fontsize=14, fontweight='bold')
plt.ylabel('Actual Species', fontsize=12)
plt.xlabel('Predicted Species', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix visualization saved as 'confusion_matrix.png'")


# ============================================================================
# 10. Visualize the Decision Tree
# ============================================================================
print("\n" + "-"*70 + "\n")

# Visualize the decision tree structure
plt.figure(figsize=(20, 10))

# Plot the tree with detailed information
plot_tree(
    dt_classifier,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,              # Color nodes by class
    rounded=True,             # Rounded box corners
    fontsize=10,
    proportion=True           # Show proportions instead of counts
)

plt.title('Decision Tree Visualization - Iris Species Classification', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
print("✓ Decision tree visualization saved as 'decision_tree.png'")

print("\nHow to read the tree:")
print("  - Each box (node) shows:")
print("    * The decision rule (e.g., 'petal width <= 0.8')")
print("    * gini: Impurity measure (0 = pure, higher = more mixed)")
print("    * samples: Proportion of samples reaching this node")
print("    * value: Distribution of samples across classes")
print("    * class: The majority class at this node")
print("  - Colors represent the dominant class at each node")
print("  - The tree splits samples based on feature values")
print("  - Leaf nodes (bottom) contain the final predictions")


# ============================================================================
# 11. Feature Importance
# ============================================================================
print("\n" + "-"*70 + "\n")

# Get feature importances from the trained model
feature_importance = dt_classifier.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("FEATURE IMPORTANCE RANKING:")
print("="*70)
print(importance_df.to_string(index=False))

print("\n→ Interpretation:")
print("  - Higher values indicate more important features for classification")
print("  - These scores show how much each feature contributes to reducing impurity")
print(f"  - The most important feature is: {importance_df.iloc[0]['Feature']}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', 
             ha='left', va='center', fontsize=10, fontweight='bold')

plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Feature Importance in Decision Tree Classifier', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance visualization saved as 'feature_importance.png'")


# ============================================================================
# 12. Summary
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70 + "\n")

print("✓ Key Findings:")
print("\n1. Data Preprocessing:")
print("   - The Iris dataset was clean with no missing values")
print("   - Labels were already encoded (0, 1, 2)")
print("   - Dataset contains 150 samples with 4 features")

print("\n2. Model Performance:")
print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   - Precision: {precision:.4f}")
print(f"   - Recall: {recall:.4f}")
print(f"   - F1-Score: {f1:.4f}")

print("\n3. Feature Importance:")
print(f"   - Most important: {importance_df.iloc[0]['Feature']}")
print(f"   - Second: {importance_df.iloc[1]['Feature']}")

print("\n4. Visualizations Created:")
print("   - confusion_matrix.png")
print("   - decision_tree.png")
print("   - feature_importance.png")

print("\n" + "-"*70)
print("\n✓ Next Steps:")
print("   - Try other classifiers (Random Forest, SVM, KNN)")
print("   - Tune hyperparameters for better performance")
print("   - Use cross-validation for more robust evaluation")
print("   - Implement feature scaling for other algorithms")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

# Show all plots
plt.show()
