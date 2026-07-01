import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class XGBoostTree:
    """A single decision tree built using XGBoost's exact math criteria."""
    def __init__(self, max_depth=6, reg_lambda=1.0, min_child_weight=0.1):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        
        # Tree structure placeholders
        self.feature_idx = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.leaf_value = None

    def fit(self, X, g, h, current_depth=0):
        # 1. Base case: if max depth is reached, turn into a terminal leaf
        if current_depth >= self.max_depth:
            self.leaf_value = -np.sum(g) / (np.sum(h) + self.reg_lambda)
            return self

        # 2. Calculate baseline score for this node
        sum_G_head = np.sum(g)
        sum_H_head = np.sum(h)
        score_head = (sum_G_head ** 2) / (sum_H_head + self.reg_lambda)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        
        num_features = X.shape[1]       #Number of features to consider for splits

        # 3. Find the optimal feature and threshold split
        for j in range(num_features):
            feature_values = X[:, j]
            unique_values = np.unique(feature_values)           #Remove duplicates and sort the feature values to consider as potential split points    

            for threshold in unique_values:
                left_mask = feature_values < threshold
                right_mask = ~left_mask                         # Reamining samples go to the right child includding missing values

                if not np.any(left_mask) or not np.any(right_mask):   # Skip splits that result in empty child nodes
                    continue

                sum_G_l, sum_H_l = np.sum(g[left_mask]), np.sum(h[left_mask])
                sum_G_r, sum_H_r = np.sum(g[right_mask]), np.sum(h[right_mask])

                # Enforce min_child_weight constraint (skip splits when child nodes are too small)
                if sum_H_l < self.min_child_weight or sum_H_r < self.min_child_weight:
                    continue

                score_left = (sum_G_l ** 2) / (sum_H_l + self.reg_lambda)
                score_right = (sum_G_r ** 2) / (sum_H_r + self.reg_lambda)
                
                gain = score_left + score_right - score_head

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = threshold

        # 4. If a valid split is found, create child branches. Otherwise, create a leaf.
        if best_gain > 0:
            self.feature_idx = best_feature
            self.threshold = best_threshold

            left_mask = X[:, self.feature_idx] < self.threshold
            right_mask = ~left_mask

            self.left_child = XGBoostTree(self.max_depth, self.reg_lambda, self.min_child_weight)
            self.right_child = XGBoostTree(self.max_depth, self.reg_lambda, self.min_child_weight)

            self.left_child.fit(X[left_mask], g[left_mask], h[left_mask], current_depth + 1)
            self.right_child.fit(X[right_mask], g[right_mask], h[right_mask], current_depth + 1)
        else:
            self.leaf_value = -sum_G_head / (sum_H_head + self.reg_lambda)

        return self

    def predict_row(self, row):
        """Recursively route a single row of data down the decision tree."""
        if self.leaf_value is not None:
            return self.leaf_value
        
        if row[self.feature_idx] < self.threshold:
            return self.left_child.predict_row(row)
        else:
            return self.right_child.predict_row(row)


class CustomXGBRegressor:
    """The main boosting controller handling sequential ensemble updates."""
    def __init__(self, n_estimators=100, learning_rate=0.05, max_depth=3, reg_lambda=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = []
        self.base_pred = None

    def fit(self, X, y):
        # Initialize predictions with the global target mean
        self.base_pred = np.mean(y)
        current_preds = np.full_like(y, self.base_pred, dtype=float)

        for i in range(self.n_estimators):
            # Calculate gradients and hessians (MSE Loss derivatives)
            # g = pred - true, h = 1
            g = current_preds - y
            h = np.ones_like(y)

            # Initialize, build, and store a new sequential tree
            tree = XGBoostTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
            tree.fit(X, g, h)
            self.trees.append(tree)

            # Generate predictions from this specific tree
            tree_preds = np.array([tree.predict_row(row) for row in X])

            # Update ensemble predictions using the learning rate step
            current_preds += self.learning_rate * tree_preds

    def predict(self, X):
        # Start predictions from the initial baseline mean value to the target
        ensemble_preds = np.full(X.shape[0], self.base_pred, dtype=float)
        
        # Aggregate the learning rate-scaled contributions from all trees
        for tree in self.trees:
            tree_preds = np.array([tree.predict_row(row) for row in X])
            ensemble_preds += self.learning_rate * tree_preds
            
        return ensemble_preds
    





df = pd.read_excel("V3.xlsx")

# Feature columns
feature_columns = [
    'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
    'Pb 82', 'Sn 50', 'Ti 22',
    'T5 ?', 'T6 ?', 'T7 ?',
    'sigma_a'
]

# Prepare data
X = df[feature_columns].values.astype(float)
y = np.log10(df['N'].values.astype(float))


print(f"Dataset shape: {X.shape}")

# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {X_train.shape}, Test: {X_test.shape}")


# 1. Initialize our scratch-built model
my_model = CustomXGBRegressor(n_estimators=150, learning_rate=0.075, max_depth=6)

# 2. Train the model
print("Training custom XGBoost model from scratch...")
my_model.fit(X_train, y_train)

# 3. Make predictions
y_pred = my_model.predict(X_test)

# 4. Measure performance
from sklearn.metrics import r2_score, mean_absolute_error
print(f"Scratch Model R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Scratch Model MAE:      {mean_absolute_error(y_test, y_pred):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True log10(N)') 
plt.ylabel('Predicted log10(N)')
plt.title('XGBoost: True vs Predicted log10(N), R² = {:.3f}'.format(r2_score(y_test, y_pred)))
plt.show()



