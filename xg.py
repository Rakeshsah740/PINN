
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class XGBoostFromScratch(BaseEstimator, RegressorMixin):
    """
    XGBoost Regressor from Scratch
    Compatible with sklearn interface (fit, predict, score)
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 lambda_reg=1.0, gamma=0, min_child_weight=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.trees = []
        self.mean_y = None
    
    def fit(self, X, y):
        """Train the model"""
        X = np.array(X)
        y = np.array(y)
        n_samples = len(y)
        
        # Initialize predictions with mean
        self.mean_y = np.mean(y)
        predictions = np.full(n_samples, self.mean_y)
        
        # Build each tree
        for tree_idx in range(self.n_estimators):
            # Calculate gradients (g) and hessians (h)
            g = y - predictions          # First derivative
            h = np.ones(n_samples)       # Second derivative (fixed at 1)
            
            # Build tree
            tree = self._build_tree(X, g, h, depth=0)
            self.trees.append(tree)
            
            # Get leaf predictions
            leaf_pred = self._predict_tree(X, tree)
            
            # Update predictions
            predictions += self.learning_rate * leaf_pred
            
            # Optional: print progress
            if (tree_idx + 1) % 50 == 0:
                mse = np.mean((y - predictions) ** 2)
                print(f"Tree {tree_idx + 1}/{self.n_estimators}, MSE: {mse:.6f}")
        
        return self
    
    def _build_tree(self, X, g, h, depth):
        """Recursively build a single tree"""
        n_samples = len(X)
        
        # Stop if max depth reached
        if depth >= self.max_depth:
            leaf_value = np.sum(g) / (np.sum(h) + self.lambda_reg)
            return ('leaf', -leaf_value)
        
        # Current node score
        sum_g_head = np.sum(g)
        sum_h_head = np.sum(h)
        score_head = (sum_g_head ** 2) / (sum_h_head + self.lambda_reg)
        
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
        # Try all features
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try all thresholds
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold
                
                # Skip if split is too small
                if np.sum(left_mask) < self.min_child_weight or np.sum(right_mask) < self.min_child_weight:
                    continue
                
                # Calculate left child score
                g_left = np.sum(g[left_mask])
                h_left = np.sum(h[left_mask])
                score_left = (g_left ** 2) / (h_left + self.lambda_reg)
                
                # Calculate right child score
                g_right = np.sum(g[right_mask])
                h_right = np.sum(h[right_mask])
                score_right = (g_right ** 2) / (h_right + self.lambda_reg)
                
                # Calculate gain
                gain = score_left + score_right - score_head - self.gamma
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
        
        # If no good split, create leaf
        if best_gain <= 0 or best_feature is None:
            leaf_value = np.sum(g) / (np.sum(h) + self.lambda_reg)
            return ('leaf', -leaf_value)
        
        # Build children recursively
        left_child = self._build_tree(
            X[best_left_mask], g[best_left_mask], h[best_left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[best_right_mask], g[best_right_mask], h[best_right_mask], depth + 1
        )
        
        return ('node', best_feature, best_threshold, left_child, right_child)
    
    def _predict_tree(self, X, tree):
        """Predict using a single tree"""
        X = np.array(X)
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            node = tree
            while node[0] != 'leaf':
                _, feature_idx, threshold, left, right = node
                if X[i, feature_idx] <= threshold:
                    node = left
                else:
                    node = right
            predictions[i] = node[1]
        
        return predictions
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        predictions = np.full(len(X), self.mean_y)
        
        for tree in self.trees:
            predictions += self.learning_rate * self._predict_tree(X, tree)
        
        return predictions
    
    def score(self, X, y):
        """R² score (sklearn compatibility)"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))