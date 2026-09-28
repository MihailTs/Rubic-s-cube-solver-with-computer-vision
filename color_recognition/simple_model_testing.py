import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def load_images_to_array(dataloader, device=None):
    """Convert images from dataloader to numpy array and labels"""
    images_list = []
    labels_list = []
    
    for images, labels in dataloader:
        if device:
            images = images.to(device)
        batch_size = images.size(0)
        images_flat = images.view(batch_size, -1)
        images_list.append(images_flat.cpu().numpy())
        labels_list.append(labels.numpy())
    
    X = np.vstack(images_list)
    y = np.concatenate(labels_list)
    return X, y

def train_sklearn_models(X_train, y_train, X_val, y_val):
    """Train KNN, SVM, Logistic Regression, and Decision Tree models with GridSearchCV"""
    results = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # KNN GridSearch
    print("\nTraining KNN with GridSearchCV...")
    knn_params = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    knn_grid = GridSearchCV(knn, knn_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    knn_grid.fit(X_train_scaled, y_train)
    best_knn = knn_grid.best_estimator_
    y_pred_knn = best_knn.predict(X_val_scaled)
    knn_accuracy = np.mean(y_pred_knn == y_val)
    knn_f1 = f1_score(y_val, y_pred_knn, average='weighted')
    results['KNN'] = (knn_accuracy, knn_f1)
    print(f"KNN Best Parameters: {knn_grid.best_params_}")
    print(f"KNN Validation Accuracy: {knn_accuracy:.4f}, F1 Score: {knn_f1:.4f}")
    
    # SVM GridSearch
    print("\nTraining SVM with GridSearchCV...")
    svm_params = {
        'kernel': ['rbf', 'poly'],
        'C': [0.1, 0.3, 0.5, 1, 3, 5, 10],
        'gamma': ['scale', 'auto']
    }
    svm = SVC()
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    svm_grid.fit(X_train_scaled, y_train)
    best_svm = svm_grid.best_estimator_
    y_pred_svm = best_svm.predict(X_val_scaled)
    svm_accuracy = np.mean(y_pred_svm == y_val)
    svm_f1 = f1_score(y_val, y_pred_svm, average='weighted')
    results['SVM'] = (svm_accuracy, svm_f1)
    print(f"SVM Best Parameters: {svm_grid.best_params_}")
    print(f"SVM Validation Accuracy: {svm_accuracy:.4f}, F1 Score: {svm_f1:.4f}")
    
    # Logistic Regression GridSearch
    print("\nTraining Logistic Regression with GridSearchCV...")
    lr_params = {
        'C': [0.1, 0.3, 0.5, 1, 5, 10, 100],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [1000, 2000]
    }
    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    lr_grid.fit(X_train_scaled, y_train)
    best_lr = lr_grid.best_estimator_
    y_pred_lr = best_lr.predict(X_val_scaled)
    lr_accuracy = np.mean(y_pred_lr == y_val)
    lr_f1 = f1_score(y_val, y_pred_lr, average='weighted')
    results['Logistic Regression'] = (lr_accuracy, lr_f1)
    print(f"Logistic Regression Best Parameters: {lr_grid.best_params_}")
    print(f"Logistic Regression Validation Accuracy: {lr_accuracy:.4f}, F1 Score: {lr_f1:.4f}")
    
    # Decision Tree GridSearch
    print("\nTraining Decision Tree with GridSearchCV...")
    dt_params = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None]
    }
    dt = DecisionTreeClassifier(random_state=42)
    dt_grid = GridSearchCV(dt, dt_params, cv=5, scoring='f1_weighted', n_jobs=-1)
    dt_grid.fit(X_train_scaled, y_train)
    best_dt = dt_grid.best_estimator_
    y_pred_dt = best_dt.predict(X_val_scaled)
    dt_accuracy = np.mean(y_pred_dt == y_val)
    dt_f1 = f1_score(y_val, y_pred_dt, average='weighted')
    results['Decision Tree'] = (dt_accuracy, dt_f1)
    print(f"Decision Tree Best Parameters: {dt_grid.best_params_}")
    print(f"Decision Tree Validation Accuracy: {dt_accuracy:.4f}, F1 Score: {dt_f1:.4f}")

    return results, scaler, best_knn, best_svm, best_lr, best_dt

def evaluate_sklearn_on_test(knn, svm, lr, dt, X_test, y_test, scaler):
    """Evaluate sklearn models on test set"""
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # KNN Test
    y_pred = knn.predict(X_test_scaled)
    knn_test_accuracy = np.mean(y_pred == y_test)
    knn_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['KNN'] = (knn_test_accuracy, knn_test_f1)
    print(f"KNN Test Accuracy: {knn_test_accuracy:.4f}, F1 Score: {knn_test_f1:.4f}")
    
    # SVM Test
    y_pred = svm.predict(X_test_scaled)
    svm_test_accuracy = np.mean(y_pred == y_test)
    svm_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['SVM'] = (svm_test_accuracy, svm_test_f1)
    print(f"SVM Test Accuracy: {svm_test_accuracy:.4f}, F1 Score: {svm_test_f1:.4f}")
    
    # Logistic Regression Test
    y_pred = lr.predict(X_test_scaled)
    lr_test_accuracy = np.mean(y_pred == y_test)
    lr_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['Logistic Regression'] = (lr_test_accuracy, lr_test_f1)
    print(f"Logistic Regression Test Accuracy: {lr_test_accuracy:.4f}, F1 Score: {lr_test_f1:.4f}")
    
    # Decision Tree Test
    y_pred = dt.predict(X_test_scaled)
    dt_test_accuracy = np.mean(y_pred == y_test)
    dt_test_f1 = f1_score(y_test, y_pred, average='weighted')
    results['Decision Tree'] = (dt_test_accuracy, dt_test_f1)
    print(f"Decision Tree Test Accuracy: {dt_test_accuracy:.4f}, F1 Score: {dt_test_f1:.4f}")
    
    return results