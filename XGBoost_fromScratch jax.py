from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split	




# -------------------------------------------------------
# TREE
# -------------------------------------------------------

class XGBoostTree:
    """A single decision tree built using XGBoost math criteria (JAX version)."""

    def __init__(self, max_depth=6, reg_lambda=1.0, min_child_weight=0.1):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight

        self.feature_idx = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.leaf_value = None

    def fit(self, X, g, h, current_depth=0):

        # 1. stop condition
        if current_depth >= self.max_depth:
            self.leaf_value = float(-jnp.sum(g) / (jnp.sum(h) + self.reg_lambda))
            return self

        # 2. node stats
        sum_G_head = float(jnp.sum(g))
        sum_H_head = float(jnp.sum(h))
        score_head = (sum_G_head ** 2) / (sum_H_head + self.reg_lambda)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        num_features = X.shape[1]

        # 3. search best split
        for j in range(num_features):

            feature_values = X[:, j]
            unique_values = np.asarray(jnp.unique(feature_values))

            for threshold in unique_values:

                left_mask = feature_values < threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                sum_G_l = float(jnp.sum(g[left_mask]))
                sum_H_l = float(jnp.sum(h[left_mask]))

                sum_G_r = float(jnp.sum(g[right_mask]))
                sum_H_r = float(jnp.sum(h[right_mask]))

                if (sum_H_l < self.min_child_weight or
                        sum_H_r < self.min_child_weight):
                    continue

                score_left = (sum_G_l ** 2) / (sum_H_l + self.reg_lambda)
                score_right = (sum_G_r ** 2) / (sum_H_r + self.reg_lambda)

                gain = float(score_left + score_right - score_head)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = float(threshold)

        # 4. split or leaf
        if best_gain > 0:
            self.feature_idx = best_feature
            self.threshold = best_threshold

            left_mask = X[:, self.feature_idx] < self.threshold
            right_mask = ~left_mask

            self.left_child = XGBoostTree(
                self.max_depth,
                self.reg_lambda,
                self.min_child_weight,
            )

            self.right_child = XGBoostTree(
                self.max_depth,
                self.reg_lambda,
                self.min_child_weight,
            )

            self.left_child.fit(
                X[left_mask],
                g[left_mask],
                h[left_mask],
                current_depth + 1,
            )

            self.right_child.fit(
                X[right_mask],
                g[right_mask],
                h[right_mask],
                current_depth + 1,
            )

        else:
            self.leaf_value = float(-sum_G_head / (sum_H_head + self.reg_lambda))

        return self

    def predict_row(self, row):

        if self.leaf_value is not None:
            return self.leaf_value

        if row[self.feature_idx] < self.threshold:
            return self.left_child.predict_row(row)
        else:
            return self.right_child.predict_row(row)


# -------------------------------------------------------
# BOOSTING CONTROLLER
# -------------------------------------------------------

class CustomXGBRegressor:

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        reg_lambda=1.0,
        min_child_weight=0.1,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight

        self.trees = []
        self.base_pred = None

    def fit(self, X, y):

        X = jnp.asarray(X)
        y = jnp.asarray(y)

        self.base_pred = float(jnp.mean(y))

        current_preds = jnp.full_like(y, self.base_pred, dtype=jnp.float32)

        self.trees = []

        for i in range(self.n_estimators):

            # gradients / hessians
            g = current_preds - y
            h = jnp.ones_like(y)

            # train tree
            tree = XGBoostTree(
                max_depth=self.max_depth,
                reg_lambda=self.reg_lambda,
                min_child_weight=self.min_child_weight,
            )

            tree.fit(X, g, h)
            self.trees.append(tree)

            # predictions from tree
            tree_preds = jnp.array([
                tree.predict_row(row) for row in X
            ], dtype=jnp.float32)

            current_preds = current_preds + self.learning_rate * tree_preds

    def predict(self, X):

        X = jnp.asarray(X)

        ensemble_preds = jnp.full(
            X.shape[0],
            self.base_pred,
            dtype=jnp.float32,
        )

        for tree in self.trees:

            tree_preds = jnp.array([
                tree.predict_row(row) for row in X
            ], dtype=jnp.float32)

            ensemble_preds += self.learning_rate * tree_preds

        return ensemble_preds


# -----------------------------------------------------------
# Example
# -----------------------------------------------------------

if __name__ == "__main__":

    df = pd.read_excel("V3.xlsx")

    # Feature columns
    feature_columns = [
        'Al 26','Si 14', 'Fe 26', 'Cu 29', 'Mn 25', 'Mg 12', 'Cr 24', 'Ni 28', 'Zn 30',
        'Pb 82', 'Sn 50', 'Ti 22',
        'T5 ?', 'T6 ?', 'T7 ?',
        'sigma_a'
    ]

    # Prepare data
    X = jnp.asarray(
    df[feature_columns].values,
    dtype=jnp.float32
    )

    y = jnp.log10(
        jnp.asarray(
            df["N"].values,
            dtype=jnp.float32
        )
    )


    print(f"Dataset shape: {X.shape}")

    # ============================================================
    # 2. TRAIN-TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training: {X_train.shape}, Test: {X_test.shape}")
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