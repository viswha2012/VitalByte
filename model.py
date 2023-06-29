import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv(r'dataset.csv')

# Adjust the label values
data['obesity_level'] = data['obesity_level'] - 1

# Verify the updated label encoding
unique_labels = data['obesity_level'].unique()
print(unique_labels)

# Separate the features and the target variable
features = data.drop('obesity_level', axis=1)
target = data['obesity_level']

# Convert categorical variables to one-hot encoding
features = pd.get_dummies(features, columns=['gender', 'family_history_with_overweight', 'caloric_food',
                                              'smoke', 'calories', 'transportation'])
											 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))  # Output layer with 7 classes for obesity levels

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy:', accuracy)

# Make predictions
predictions = model.predict(X_test[0:1])
predicted_labels = np.argmax(predictions, axis=1)

# Save the model
model.save('obesity_model.h5')									 
pickle.dump(scaler, open('scaler.pkl', 'wb'))
