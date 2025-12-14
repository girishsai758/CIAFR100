import setuptools
import os
import re
import string
import pandas as pd
import mlflow
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import mlflow.sklearn
import dagshub
import mlflow.entities # For logging parameters
import warnings
import torchvision
import torchvision.models as models
import yaml
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, recall_score, precision_score
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import mean_squared_error
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
#setting up mlflow and dagshub
CONFIG = {
    
    "mlflow_tracking_uri": "https://dagshub.com/girishsai758/CIAFR100.mlflow",
    "dagshub_repo_owner": "girishsai758",
    "dagshub_repo_name": "CIAFR100",
    "experiment_name": "exp2"
}
# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# Fetch pre-trained MobileNet V2 model
mobilenet_v2 = models.mobilenet_v2(pretrained=True)
print("MobileNet V2 pre-trained model fetched successfully.")
# You can inspect the model architecture by printing it
# print(mobilenet_v2)

# Ensure feature layers are trainable (i.e., use their parameters for training)
for param in mobilenet_v2.features.parameters():
    param.requires_grad = False
print("Feature layers are freezed: parameters will  not be updated during training.")

# Define transformations for the test images
# MobileNetV2 expects 224x224 input, and ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1. Load the CIFAR-100 training dataset
try:
    train_data = datasets.CIFAR100(
        root='./cifar_data',
        download=True,
        train=True,
        transform=transform
    )

    # 2. Create DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True
    )

    print("--- TRAINING DATA LOADING SUCCESSFUL ---")
    print(f"Total training images loaded: {len(train_data)}")

    # 3. Load the CIFAR-100 testing dataset
    test_data = datasets.CIFAR100(
        root='./cifar_data',
        download=True,
        train=False,
        transform=transform
    )

    # 4. Create DataLoader for the testing set
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=False
    )

    print("--- TESTING DATA LOADING SUCCESSFUL ---")
    print(f"Total testing images loaded: {len(test_data)}")

    # Verification of a batch (optional, but good for sanity check)
    # images_train, labels_train = next(iter(train_loader))
    # print(f"\nShape of one batch of transformed training images: {images_train.shape}")
    # print("Labels of one training batch (0-99 integers):", labels_train)

    # images_test, labels_test = next(iter(test_loader))
    # print(f"\nShape of one batch of transformed testing images: {images_test.shape}")
    # print("Labels of one testing batch (0-99 integers):", labels_test)

except Exception as e:
    print(f"An error occurred during data loading: {e}")

print("Data loading and preprocessing completed.")

learning_rates = [0.001, 0.01, 0.1]
epochs_list = [100, 150, 200]
print(f"Defined learning rates: {learning_rates}")
print(f"Defined epochs lists: {epochs_list}")

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_test_loss = running_loss / len(test_loader.dataset)
    accuracy = correct_predictions / total_predictions

    # Calculate F1-score, recall, and precision
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)

    return avg_test_loss, accuracy, f1, recall, precision

# Keep an original copy of the model state dictionary
original_mobilenet_v2_state_dict = copy.deepcopy(mobilenet_v2.state_dict())
# Load parameters from params.yaml
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

classifier_params = params['classifier_params']
layer_sizes = classifier_params['layer_sizes']
dropout_rates = classifier_params['dropout_rates']
results_list = []
# Iterate through each combination of learning rates and epochs
for j in range(0,9):
 for k in range(0,1):
  for lr in learning_rates:
    optimizer = optim.Adam(mobilenet_v2.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for num_epochs in epochs_list:
        PATIENCE = 15 # Wait 15 epochs after no improvement
        MIN_DELTA = 1e-4 # Minimum improvement to be considered significant

        best_loss = float('inf')
        patience_counter = 0
        stopped_epoch = num_epochs
        trial_metrics = {
        
        'learning_rate': lr,
        'final_test_loss': None,
        'best_test_accuracy': 0.0,
        'best_acc_epoch': 0,
        'stopped_epoch': num_epochs,
        'dropoutrate': dropout_rates[0]*0.1*j,
        'layer_size1':layer_sizes[0],
        'layer_size2':(layer_sizes[1]/2**k),
        'layer_size3':(layer_sizes[2]/2**k),
        'layer_size4':layer_sizes[3]
    }
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_param('learning_rate', lr)
            mlflow.log_param('epochs', num_epochs)
            mlflow.log_param('dropoutrate', dropout_rates[0]*0.1*j)
            mlflow.log_param('layer_size1',layer_sizes[0])
            mlflow.log_param('layer_size2',(layer_sizes[1]/2**k))
            mlflow.log_param('layer_size3',(layer_sizes[2]/2**k))
            mlflow.log_param('layer_size4',layer_sizes[3])

            # Re-initialize mobilenet_v2 to its original pre-trained state
            mobilenet_v2.load_state_dict(original_mobilenet_v2_state_dict)
            mobilenet_v2 = mobilenet_v2.to(device)

            # Freeze the feature layers of the re-initialized mobilenet_v2 model
            for param in mobilenet_v2.features.parameters():
                param.requires_grad = False

            # Reconstruct the classifier dynamically
            layers = []
            input_features = layer_sizes[0]
            for i in range(len(layer_sizes) - 1):

                output_features = int(layer_sizes[i + 1]/2**k)
                layers.append(nn.Linear(input_features, output_features))
                if i < len(dropout_rates):
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rates[i]*0.1*j))
                elif i < len(layer_sizes) - 1:
                    layers.append(nn.ReLU())
                input_features = output_features
            mobilenet_v2.classifier = nn.Sequential(*layers).to(device)

            # Log the classifier architecture
            mlflow.log_param('classifier_architecture', str(mobilenet_v2.classifier))



            print(f"\nStarting training for LR: {lr}, Epochs: {num_epochs}")
            for epoch in range(num_epochs):
                train_loss = train_model(mobilenet_v2, train_loader, criterion, optimizer, device)
                # --- Early Stopping Logic ---

                test_loss, test_accuracy,f1,recall,precision = evaluate_model(mobilenet_v2, test_loader, criterion, device)
                if test_accuracy > best_acc:
                  best_acc = test_accuracy
                  best_epoch_acc = epoch + 1
            # --- CRITICAL: Save the BEST model weights here using best_acc as reference ---
            # save_best_model(model, trial_name, metric='accuracy')
                # --- Early Stopping Logic ---
                if test_loss < best_loss - MIN_DELTA:
                 # Improvement is significant: reset counter and update best loss
                 best_loss = test_loss
                 patience_counter = 0
            # --- CRITICAL: Save the BEST model weights here ---
            # save_best_model(model, trial_name)

                else:
            # No significant improvement: increment counter
                 patience_counter += 1

                print(f" Epoch {epoch+1}/{num_epochs}, Test Loss: {test_loss:.6f} (Best: {best_loss:.6f}, Patience: {patience_counter}/{PATIENCE})")


                if patience_counter >= PATIENCE:
                  stopped_epoch = epoch + 1
                  print(f"\n!!!  Early stopping triggered at epoch {stopped_epoch}. Validation loss hasn't improved for {PATIENCE} epochs.")
                  break

                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

                # mlflow.log_metric(f"train_loss_epoch", train_loss, step=epoch)
                # mlflow.log_metric(f"test_loss_epoch", test_loss, step=epoch)
                # mlflow.log_metric(f"test_accuracy_epoch", test_accuracy, step=epoch)

            # Log final metrics for the run
            mlflow.log_metric("final_test_loss", test_loss)
            mlflow.log_metric("final_test_accuracy", test_accuracy)
            mlflow.log_metric("final_test_f1", f1)
            mlflow.log_metric("final_test_recall", recall)
            mlflow.log_metric("final_test_precision", precision)
            mlflow.log_param("stopped_epoch", stopped_epoch)
            trial_metrics['final_test_loss'] = test_loss # The loss when training stopped
            trial_metrics['best_test_accuracy'] = best_acc
            trial_metrics['best_acc_epoch'] = best_epoch_acc
            trial_metrics['stopped_epoch'] = stopped_epoch
    
            results_list.append(trial_metrics)
            # # Log the trained model
            # mlflow.pytorch.log_model(mobilenet_v2, "model")

            mlflow.end_run()

print("MLflow experiment loop completed, models and metrics logged.")
def display_results(results):
    """Prints the summary table and identifies the best trial."""
    if not results:
        print("\nNo results collected.")
        return

    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY")
    print("="*80)
    
    # Convert Manager list to regular list for sorting
    results_list = list(results)
    
    # Find the best trial based on best_test_accuracy
    best_trial = max(results_list, key=lambda x: x['best_test_accuracy'])

    # Print Header
    print(f" {'LR':<10} | {'Best Acc':<10} | {'Acc Epoch':<10} | {'Final Loss':<12} | {'Stopped @':<10}")
    print("-" * 80)
    
    
    

    print("-" * 80)
    print(f"OVERALL BEST MODEL: (Accuracy: {best_trial['best_test_accuracy']:.4f}) | LR: {best_trial['learning_rate']}, Stopped at Epoch: {best_trial['stopped_epoch']}|  Acc Epoch: {best_trial['best_acc_epoch']} | dropoutrate: {best_trial['dropoutrate']}, layer_size1: {best_trial['layer_size1']}, layer_size2: {best_trial['layer_size2']}, layer_size3: {best_trial['layer_size3']}, layer_size4: {best_trial['layer_size4']}")
    
    print("="*80)
display_results(results_list)


