import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

# read model from csv
with open('model.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile, delimiter='\t'))

storyPointsData = [float(item[0]) for item in data]
timeHoursData   = [float(item[1]) for item in data]

print("storyPointsData: ", storyPointsData)
print("timeHoursData: ", timeHoursData)

# Define X and y
# Usamos el reshape para el coeficiente de Pearson
storyPoints = np.array(storyPointsData).reshape([-1,1])
timeHours   = np.array(timeHoursData)

# Instance model
model = LinearRegression()

# Train model
model.fit(storyPoints, timeHours)

# find b1
intercept = model.intercept_
print('y = b1X + b0')
print(f'b0: {intercept:.4f}')

# find b0
slope = model.coef_
print(f'b1: {slope.round(4)}')

# now
print(f"y = {intercept} + {slope} * X")
# print('Hello!')

# Predict with manual model
y_pred_eq = intercept + slope * storyPoints
print(y_pred_eq.tolist())

r_squared = model.score(storyPoints, timeHours)
print(f'Coeficiente de determinación: {r_squared}')

# Area del plot
plt.figure(figsize=(8,6))

# Gráfico
plt.scatter(storyPoints, timeHours,  color='blue')

# Acá puedo utilizar el predict o el modelo manual
plt.plot(storyPoints, model.predict(storyPoints), color='red', linewidth=2)

# Defino el valor y rango de los ejes
plt.xticks(np.arange(1,12))
plt.yticks(np.arange(0,18))

# xy labels
plt.xlabel("Story Points")
plt.ylabel("Time (h)")

# Titulo del graficop
plt.title("Performance: Team booster", fontweight="bold", size=15)

# plot
plt.show()