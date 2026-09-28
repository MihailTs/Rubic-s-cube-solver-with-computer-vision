import joblib
import pandas as pd
from sklearn.metrics import f1_score
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

from color_cnn import ColorCNN
from simple_model_testing import evaluate_sklearn_on_test, load_images_to_array, train_sklearn_models


def main():

    train_transforms = transforms.Compose(
        [transforms.Resize((20, 20)),
         transforms.ToTensor()])

    dataset_train = ImageFolder(
        'data/colors_train',
        transform=train_transforms,
    )

    dataset_test = ImageFolder(
        'data/colors_test',
        transform=train_transforms,
    )

    dataset_validate = ImageFolder(
        'data/colors_validation',
        transform=train_transforms,
    )

    dataloader_train = data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=32,
    )

    dataloader_test = data.DataLoader(
        dataset_test,
        shuffle=True,
        batch_size=32,
    )

    dataloader_validate = data.DataLoader(
        dataset_validate,
        shuffle=True,
        batch_size=32,
    )

    # light testing multiple hypotheses
    poolings = ['max_pooling', 'average_pooling']
    kernel_sizes = [3, 5]
    activation_functions = ['relu', 'sigmoid', 'elu', 'tanh']

    num_classes = len(dataset_train.classes)

    experiments = []
    for pooling in poolings:
        for kernel_size in kernel_sizes:
            for activation_function in activation_functions:
                experiment = {}
                model = ColorCNN(num_classes,
                                 pooling=pooling,
                                 kernel_size=kernel_size,
                                 activation_function=activation_function)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # training loop
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                num_epochs = 16
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0

                    for images, labels in dataloader_train:
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
                    )

                # validation loop
                model.eval()
                correct = 0
                total = 0
                all_predicted = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in dataloader_validate:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        all_predicted.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_accuracy = 100 * correct / total
                val_f1 = f1_score(all_labels, all_predicted, average='weighted')
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                print(f"Validation F1 Score: {val_f1:.4f}")

                experiment['model'] = 'CNN'
                experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
                experiment['pooling'] = pooling
                experiment['kernel_size'] = kernel_size
                experiment['activation_function'] = activation_function
                experiment['accuracy'] = correct / total
                experiment['f1_score'] = val_f1
                experiments.append(experiment)

    kernel_sizes = [3]
    activation_functions = ['relu', 'sigmoid']

    for pooling in poolings:
        for kernel_size in kernel_sizes:
            for activation_function in activation_functions:
                experiment = {}
                model = ColorCNN(num_classes,
                                 pooling=pooling,
                                 kernel_size=kernel_size,
                                 activation_function=activation_function)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                # training loop
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)

                num_epochs = 20
                for epoch in range(num_epochs):
                    model.train()
                    running_loss = 0.0

                    for images, labels in dataloader_train:
                        images, labels = images.to(device), labels.to(device)

                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
                    )

                # validation loop
                model.eval()
                correct = 0
                total = 0
                all_predicted = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in dataloader_validate:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        all_predicted.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_accuracy = 100 * correct / total
                val_f1 = f1_score(all_labels, all_predicted, average='weighted')
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                print(f"Validation F1 Score: {val_f1:.4f}")

                experiment['model'] = 'CNN'
                experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
                experiment['pooling'] = pooling
                experiment['kernel_size'] = kernel_size
                experiment['activation_function'] = activation_function
                experiment['accuracy'] = correct / total
                experiment['f1_score'] = val_f1
                experiments.append(experiment)

    df = pd.DataFrame(experiments)
    with pd.ExcelWriter('model_report_color_prediction.xlsx', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, index=False)

    experiment = {}
    model = ColorCNN(num_classes,
                      pooling='max_pooling',
                      kernel_size=3,
                      activation_function='relu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
        )

    # validation loop
    model.eval()
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader_validate:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total
    val_f1 = f1_score(all_labels, all_predicted, average='weighted')
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1 Score: {val_f1:.4f}")

    experiment['model'] = 'CNN'
    experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
    experiment['pooling'] = 'max_pooling'
    experiment['kernel_size'] = 3
    experiment['activation_function'] = 'relu'
    experiment['accuracy'] = correct / total
    experiment['f1_score'] = val_f1

    # testing loop for the best CNN model
    model = ColorCNN(num_classes,
                      pooling='max_pooling',
                      kernel_size=3,
                      activation_function='relu')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_epochs = 16
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
        )

    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    experiment = {}
    with torch.no_grad():
        for images, labels in dataloader_test:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100 * correct / total
    val_f1 = f1_score(all_labels, all_predicted, average='weighted')
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation F1 Score: {val_f1:.4f}")

    experiment['model'] = 'CNN'
    experiment['architecture'] = 'layer 1: 16 filters; pooling; layer 2: 32 filters; pooling; layer 3: linear 64; layer 4: linear 6'
    experiment['pooling'] = 'max_pooling'
    experiment['kernel_size'] = 3
    experiment['activation_function'] = 'relu'
    experiment['accuracy'] = correct / total
    experiment['f1_score'] = val_f1
    print(experiment)

    # SKLEARN MODELS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data for sklearn models
    print("\nLoading training data...")
    X_train, y_train = load_images_to_array(dataloader_train, device)
    print(f"Training data shape: {X_train.shape}")

    print("Loading validation data...")
    X_val, y_val = load_images_to_array(dataloader_validate, device)
    print(f"Validation data shape: {X_val.shape}")

    print("Loading test data...")
    X_test, y_test = load_images_to_array(dataloader_test, device)
    print(f"Test data shape: {X_test.shape}")

    # Training sklearn models
    sklearn_val_results, scaler, knn, svm, lr, dt = train_sklearn_models(X_train, y_train, X_val, y_val)

    sklearn_test_results = evaluate_sklearn_on_test(knn, svm, lr, dt, X_test, y_test, scaler)
    print(sklearn_test_results)

    # Saving best KNN model
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    knn.fit(X_train_scaled, y_train)
    unique_labels = np.array(dataset_train.classes)
    joblib.dump(knn, 'knn_color_predictor.sav')
    joblib.dump(scaler, 'scaler_knn.sav')
    joblib.dump(unique_labels, 'label_mapping_knn.sav')

    # Saving the best CNN model
    best_cnn_model = ColorCNN(num_classes,
                     pooling="max_pooling",
                     kernel_size=3,
                     activation_function="relu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_cnn_model.parameters(), lr=0.001)

    # training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_cnn_model.to(device)

    num_epochs = 20
    for epoch in range(num_epochs):
        best_cnn_model.train()
        running_loss = 0.0

        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = best_cnn_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader_train):.4f}"
        )

    torch.save(best_cnn_model, 'cnn_color_predictor.pth')

    if hasattr(dataloader_train.dataset, 'class_to_idx'):
        label_mapping = {v: k for k, v in dataloader_train.dataset.class_to_idx.items()}
    elif hasattr(dataloader_train.dataset, 'classes'):
        label_mapping = {i: name for i, name in enumerate(dataloader_train.dataset.classes)}

    joblib.dump(label_mapping, 'label_mapping_cnn.sav')


if __name__ == "__main__":
    main()
