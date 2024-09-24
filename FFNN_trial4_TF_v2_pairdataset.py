# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:52:30 2024

@author: kenneyke
"""

import os
import sys
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
# from collections import Counter
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedStratifiedKFold
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PyQt5.QtWidgets import QApplication, QFileDialog
from openpyxl import Workbook

# Set TF_ENABLE_ONEDNN_OPTS environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#%% Stratiffied KFold Cross Validation
n_splits = 3  # Number of folds
n_repeats = 5  # Number of repetitions

#%% Functions Definitions

def print_progress(progress):
    sys.stdout.write(f"\rProgress: {progress}%")
    sys.stdout.flush()

def reclassify_labels(X_train, y_train, X_test, y_test, reclassify_vlaue):
    # Specify the reclassification label
    reclassification_label = reclassify_vlaue

    # Ensure y_train and y_test are pandas Series
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    unique_labels_train = set(y_train)
    unique_labels_test = set(y_test)

    common_labels = unique_labels_train.intersection(unique_labels_test)
    removed_values_train = unique_labels_train - common_labels
    removed_values_test = unique_labels_test - common_labels
   
    # Reclassify labels not present in both y_train and y_test
    y_train = y_train.replace(list(removed_values_train), reclassification_label)
    y_test = y_test.replace(list(removed_values_test), reclassification_label)
    
    # Update common_labels after reclassification
    common_labels = common_labels.union([reclassification_label])
    
    return X_train, y_train, X_test, y_test

def calculate_normalized_class_weights(y_train, class_label_weights):
    # Map the predefined weights to the classes found in y_train
    class_weights = {}
    encoded_class_weights = {}
    total_weight = 0
    
    # Create a mapping from class labels to weights
    weight_mapping = dict(zip(class_label_weights.iloc[:, 0], class_label_weights.iloc[:, 1]))
    
    # Assign weights based on y_train distribution and the predefined weights
    for class_label in y_train.unique():
        class_weights[class_label] = weight_mapping.get(class_label, 1)
        total_weight += class_weights[class_label]
    
    # Normalize the weights so they sum to 1
    class_weights = {k: v / total_weight for k, v in class_weights.items()}
    
    # Encode labels in class_weights
    for class_label, weight in class_weights.items():
        encoded_label = label_encoder.transform([class_label])[0]
        encoded_class_weights[encoded_label] = weight
    
    return encoded_class_weights

def objective(trial):
    # Hyperparameters to be tuned by Optuna
    numHidden1 = trial.suggest_int('numHidden1', 50, 100)
    numHidden2 = trial.suggest_int('numHidden2', 30, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    # activation_choices = ['relu', 'sigmoid']
    activation_choices = ['relu', 'sigmoid', 'tanh']
    activation_hidden1 = trial.suggest_categorical('activation_hidden1', activation_choices)
    activation_hidden2 = trial.suggest_categorical('activation_hidden2', activation_choices)
    num_epochs = trial.suggest_int('num_epochs', 100, 1000)

    # Model definition (as before)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(numHidden1, activation=activation_hidden1),
        tf.keras.layers.Dense(numHidden2, activation=activation_hidden2),
        tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
    ])

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',  # Monitor validation loss
        patience=15,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
    )
    
    best_val_accuracy = 0
    best_model = None
    
    for train_index, val_index in rskf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]    
               
        # Train the model
        history = model.fit(X_train_fold, y_train_fold, epochs=num_epochs, batch_size=32, 
                            validation_data=(X_val_fold, y_val_fold ), class_weight=class_weights, verbose=0, callbacks=[early_stopping])
        
        # Uncomment "Train the model" lines below to introduce early stopping
        # # Train the model
        # history = model.fit(X_train_fold, y_train_fold, epochs=num_epochs, batch_size=32, 
        #                     validation_data=(X_val_fold, y_val_fold), verbose=0, callbacks=[early_stopping])
        
        val_accuracy = np.max(history.history['val_accuracy'])
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save the model if it's the best found so far
            best_model = model
            
    trial.set_user_attr('best_val_accuracy', best_val_accuracy)
    # Save the best model
    if best_model:
        best_model.save('best_model.keras')    
    
    return best_val_accuracy

#%% Data Preprocessing

# Load Excel CSV pairs file
pairs_file_path = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\2HWY_Site-Site_DatasetPair.csv'
pairs_df = pd.read_csv(pairs_file_path)

# Get the directory path of the pairs file
output_dir = os.path.dirname(pairs_file_path)

# Initialize Excel workbook and sheet
excel_file_path = os.path.join(output_dir, 'NN_Site-Site_Accuracy_Matrix-Final.xlsx')
wb = Workbook()
ws = wb.active

# Lists to store accuracy scores
accuracy_scores = []
unique_training_files = pairs_df['Training'].unique()
unique_testing_files = pairs_df['Testing'].unique()

# Class weight excel path/directory
label_Weight_path = r'D:\ODOT_SPR866\My Label Data Work\New Manual Labelling\6_Analysis\Label_Class_Weights.xlsx'
class_label_weights = pd.read_excel(label_Weight_path)
reclassify_vlaue = int(input("\nEnter the reclassification value: "))
total_pairs = len(pairs_df)
for index, row in pairs_df.iterrows():
    training_file = row['Training']
    testing_file = row['Testing']
    
    # Load and preprocess data
    Train_data = pd.read_csv(training_file)
    Test_data = pd.read_csv(testing_file)
    
    # Training
    Train_cols_to_remove = ['Sub_class', 'In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 
                            'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   # Columns to exclude from the X data
    X_train = Train_data.drop(columns=Train_cols_to_remove, axis=1)  
    y_train = Train_data['Ext_class']
     
    # Testing
    Test_cols_to_remove = ['Sub_class','In_Class_Prio', 'Ext_Class_Label','File Paths', 'Ext_class %', 'Total Points', 
                           'Root_class', 'Ext_class', 'bb_centerX', 'bb_centerY', 'bb_centerZ']   # Columns to exclude from the X data
    X_test = Test_data.drop(columns=Test_cols_to_remove, axis=1)    
    y_test = Test_data['Ext_class']
    
    # reclassify labels
    X_train, y_train, X_test, y_test = reclassify_labels(X_train, y_train, X_test, y_test, reclassify_vlaue)
        
    # Label encoding
    label_encoder = LabelEncoder()
    combined_labels = pd.concat([y_train, y_test])
    label_encoder.fit(combined_labels) 
    
    # Compute Class weights for training 
    class_weights = calculate_normalized_class_weights(y_train, class_label_weights)
    
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)   # End of label encoding
    
    # Initialize the Repeated Stratified k-Fold Cross-Validation
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    # Optimally tune hyperparams for the NN
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    # Print the best parameters
    # print('Best parameters:', study.best_params)
    # print('Best validation accuracy:', study.best_value)

    # Load the best model from native Keras format
    best_model = load_model('best_model.keras')

    # Make predictions on the test set
    predictions = best_model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)  # Get the index of the max value which represents the class

    # Evaluate the model
    class_weights_list = [class_weights[label] for label in y_test]
    accuracy = accuracy_score(y_test, predictions, sample_weight=class_weights_list)
    # print(f'\n\nAccuracy: {accuracy}')

    # # Decode the predictions for classification report and confusion matrix
    # decoded_predictions = label_encoder.inverse_transform(predictions)
    # y_test_decoded = label_encoder.inverse_transform(y_test)
    
    # #Unique labels in decoded form for creating reports
    # unique_labels = np.unique(np.concatenate([y_train, y_test]))  
    # unique_labels_decoded = label_encoder.inverse_transform(unique_labels)
    # target_names = [str(label) for label in unique_labels_decoded]
    
    # print('\nClassification Report:')
    # print(classification_report(y_test_decoded, decoded_predictions, labels=unique_labels_decoded, target_names=target_names))
    
    # # Generate confusion matrix with decoded labels
    # print('\nConfusion Matrix:')
    # cm = confusion_matrix(y_test_decoded, decoded_predictions, labels=unique_labels_decoded)
    
    # # Plot confusion matrix using Seaborn
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=target_names, yticklabels=target_names)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()
    
    # Store accuracy score
    accuracy_scores.append(accuracy)
    
    # Calculate progress percentage
    progress = (index + 1) * 100 // total_pairs
    print_progress(progress)
    
#Write to excel
print("\nStarting to write out the CSV") 
ws.append([''] + [f'Test Site {i+1}' for i in range(len(unique_testing_files))])

# Iterate through each pair of training and testing files
for i, training_file in enumerate(unique_training_files):
    row = [f'Train Site {i+1}']
    for j, testing_file in enumerate(unique_testing_files):
        accuracy = accuracy_scores[i * len(unique_testing_files) + j]  # Assuming accuracy_scores is a 1D list
        row.append(accuracy)
    ws.append(row)

# Save the workbook to Excel file
wb.save(excel_file_path)

print("\nTask completed.")    