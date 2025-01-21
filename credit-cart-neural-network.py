import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.python.training.adam import AdamOptimizer

data = pd.read_csv('loan_data.csv')
print(data.head())
print("-"*100)
print(data.info())

le_gender = LabelEncoder()
data['person_gender'] = le_gender.fit_transform(data['person_gender'])
le_education = LabelEncoder()
data['person_education'] = le_education.fit_transform(data['person_education'])
data = pd.get_dummies(data, columns=['person_home_ownership'], drop_first=True)
data = pd.get_dummies(data, columns=['loan_intent'], drop_first=True)
le_defaults = LabelEncoder()
data['previous_loan_defaults_on_file'] = le_defaults.fit_transform(data['previous_loan_defaults_on_file'])


data['loan_to_income_ratio'] = data['loan_amnt'] / data['person_income']
data['credit_history_age_ratio'] = data['cb_person_cred_hist_length'] / data['person_age']
data['credit_score_age_ratio'] = data['credit_score'] / data['person_age']
data['income_education_interaction'] = data['person_income'] * data['person_education']
data['loan_to_credit_score_ratio'] = data['loan_amnt'] / data['credit_score']
data['debt_burden_ratio'] = data['loan_amnt'] / data['person_income']
data['is_young'] = (data['person_age'] < 30).astype(int)
data['has_long_credit_hist'] = (data['cb_person_cred_hist_length'] > 10).astype(int)
data['high_loan_amnt'] = (data['loan_amnt'] > data['loan_amnt'].mean()* 1.2).astype(int)
data['high_interest_rate'] = (data['loan_int_rate'] > data['loan_int_rate'].mean() * 1.2 ).astype(int)
data['income_emp_exp_interaction'] = data['person_income'] * data['person_emp_exp']

print(data.head())
print("-"*100)
print(data.info())

X = data.drop('loan_status', axis=1)
y = data['loan_status']


scaler = StandardScaler()
numerical_cols = ['person_age', 'person_income', 'person_emp_exp',
                  'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                  'cb_person_cred_hist_length', 'credit_score', 'loan_to_income_ratio',
                  'credit_history_age_ratio', 'income_education_interaction',
                  'loan_to_credit_score_ratio', 'income_emp_exp_interaction', 'debt_burden_ratio',
                  'credit_score_age_ratio']

X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)


model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
    LeakyReLU(alpha=0.005),
    Dense(16),
    LeakyReLU(alpha=0.005),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()