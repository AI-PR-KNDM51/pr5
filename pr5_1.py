import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# Відділення ознак (X) від цільової змінної (y)
X = df.drop('Delivery_Time', axis=1)
y = df['Delivery_Time']

# Розділення даних на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення моделі лінійної регресії
model = LinearRegression()
model.fit(X_train, y_train)  # Навчання моделі на навчальних даних

# Прогнозування цільової змінної на тестових даних
y_pred = model.predict(X_test)

# Обчислення середньоквадратичної помилки (MSE)
mse = mean_squared_error(y_test, y_pred)
# Обчислення середньої абсолютної помилки (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Виведення значень MSE та MAE
print('MSE:', mse)
print('MAE:', mae)

# Візуалізація реальних vs прогнозованих значень часу доставки
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Реальний час доставки')
plt.ylabel('Прогнозований час доставки')
plt.title('Реальний vs Прогнозований час доставки')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Діагональна лінія для орієнтації
plt.show()

# Обчислення кореляційної матриці
plt.figure(figsize=(12,10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Кореляційна матриця')
plt.show()

# Вибір релевантних ознак з високою кореляцією з цільовою змінною
corr_matrix = df.corr()
corr_target = corr_matrix['Delivery_Time'].abs().sort_values(ascending=False)
top_feature = corr_target.index[1]  # Вибір найрелевантнішої ознаки (крім самої цільової змінної)

# Відділення однієї релевантної ознаки для уніваріантної регресії
X_single = df[[top_feature]]
y = df['Delivery_Time']

# Розділення даних на навчальну та тестову вибірки для уніваріантної регресії
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y, test_size=0.2, random_state=42)

# Створення та навчання моделі лінійної регресії на одній ознаці
model_single = LinearRegression()
model_single.fit(X_train_s, y_train_s)

# Прогнозування на тестових даних
y_pred_s = model_single.predict(X_test_s)

# Обчислення MSE та MAE для уніваріантної регресії
mse_s = mean_squared_error(y_test_s, y_pred_s)
mae_s = mean_absolute_error(y_test_s, y_pred_s)

# Виведення результатів уніваріантної регресії
print('Univariate Linear Regression using', top_feature)
print('MSE:', mse_s)
print('MAE:', mae_s)

# Візуалізація реальних та прогнозованих значень для уніваріантної регресії
plt.figure(figsize=(10,5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Реальні значення')
plt.scatter(X_test_s, y_pred_s, color='red', label='Прогнозовані значення')
plt.xlabel(top_feature)  # Підпис осі X з назвою вибраної ознаки
plt.ylabel('Delivery Time')  # Підпис осі Y
plt.title('Уніваріантна лінійна регресія')
plt.legend()
plt.show()