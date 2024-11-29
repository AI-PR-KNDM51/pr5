import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Встановлення насіння для відтворюваності результатів
np.random.seed(42)

# Завантаження набору даних з CSV файлу
df = pd.read_csv('amazon_delivery.csv')

# Видалення рядків з пропущеними значеннями
df.dropna(inplace=True)

# Список категоріальних стовпців для кодування
categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
label_encoders = {}  # Словник для зберігання енкодерів

# Кодування категоріальних змінних за допомогою LabelEncoder
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Збереження енкодера для можливого подальшого використання

# Об'єднання дат та часу замовлення та отримання часу пікапу
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Order_Time'])
df['Pickup_DateTime'] = pd.to_datetime(df['Order_Date'] + ' ' + df['Pickup_Time'])

# Обчислення різниці часу між замовленням і пікапом у хвилинах
df['Time_Difference'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60

# Видалення непотрібних стовпців після обчислення різниці часу
df = df.drop(['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time', 'Order_DateTime', 'Pickup_DateTime'], axis=1)

# Масштабування числових ознак для нормалізації
scaler = StandardScaler()
numerical_cols = ['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
                  'Drop_Latitude', 'Drop_Longitude', 'Time_Difference']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Обчислення медіанного часу доставки
median_delivery_time = df['Delivery_Time'].median()

# Створення цільової змінної 'Delivery_Speed' на основі медіанного часу доставки
df['Delivery_Speed'] = (df['Delivery_Time'] <= median_delivery_time).astype(int)

# Видалення оригінального стовпця 'Delivery_Time' після створення цільової змінної
df = df.drop('Delivery_Time', axis=1)

# Відділення ознак (X) від цільової змінної (y)
X = df.drop('Delivery_Speed', axis=1)
y = df['Delivery_Speed']

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення моделі логістичної регресії з максимальною кількістю ітерацій 1000
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)  # Навчання моделі на навчальних даних

# Прогнозування цільової змінної на тестових даних
y_pred = model.predict(X_test)

# Обчислення точності моделі
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Створення матриці сплутаності
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_mat)

# Виведення звіту про класифікацію, включаючи precision, recall, f1-score
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Візуалізація матриці сплутаності за допомогою теплової карти
plt.figure(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')  # Підпис осі X
plt.ylabel('Actual')     # Підпис осі Y
plt.title('Confusion Matrix')  # Заголовок графіка
plt.show()

# Створення DataFrame з коефіцієнтами моделі для подальшого аналізу
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Сортування ознак за значенням коефіцієнтів
coef_df = coef_df.sort_values(by='Coefficient')

# Візуалізація коефіцієнтів моделі логістичної регресії за допомогою горизонтального стовпчикового графіка
plt.figure(figsize=(10,6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'])
plt.xlabel('Coefficient')  # Підпис осі X
plt.ylabel('Feature')      # Підпис осі Y
plt.title('Feature Coefficients in Logistic Regression')  # Заголовок графіка
plt.show()