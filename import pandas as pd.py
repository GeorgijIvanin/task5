import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
import time

data = pd.read_csv('train.csv', index_col=0)
target = data.target.values
data = data.drop('target', axis=1)

np.random.seed(910)
maskPlus = np.random.choice(np.where(target == 1)[0], 100000, replace=True)
maskZero = np.random.choice(np.where(target == 0)[0], 100000, replace=True)

data = pd.concat((data.iloc[maskPlus], data.iloc[maskZero]))
target = np.hstack((target[maskPlus], target[maskZero]))

xTrain, xTest, yTrain, yTest = train_test_split(data, target, test_size=0.5)

scaler = StandardScaler()

#Выберем только числовые колонки и заполним пустоты медианой и пропуски пустотой
catCols = [col for col in data.columns if 'cat' in col]
binCols = [col for col in data.columns if 'bin' in col]
numericCols = [col for col in data.columns if col not in catCols + binCols]

xTrain_1 = xTrain[numericCols].copy()
xTest_1 = xTest[numericCols].copy()

xTrain_1 = xTrain_1.replace(-1, np.nan)
xTest_1 = xTest_1.replace(-1, np.nan)

xTrain_1 = xTrain_1.fillna(xTrain_1.median())
xTest_1 = xTest_1.fillna(xTrain_1.median())

xTrain_1= scaler.fit_transform(xTrain_1)
xTest_1 = scaler.transform(xTest_1)


#Находим через GridSearchCV параметры для поиска оптимума
paramGrid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear'] #Работает с обеими регуляризациями
}

gridSearch = GridSearchCV(LogisticRegression(max_iter=1000), paramGrid, cv=3, n_jobs=-1, verbose=1)
gridSearch.fit(xTrain_1, yTrain)
bestLogReg = gridSearch.best_estimator_

#Обучаем
yPredOptimum = bestLogReg.predict(xTest_1)
yPredProbaOptimum = bestLogReg.predict_proba(xTest_1)[:, 1]
accuracyOptimum = accuracy_score(yTest, yPredOptimum)

print(classification_report(yTest, yPredOptimum))





#Функция для обучения с отслеживанием потерь на каждой итерации
def trainLogisticRegressionWithTracking(c, xTrain, xTest, yTrain, yTest, max_iter=100):
    trainLosses = []
    testLosses = []

    model = LogisticRegression(
        C=c, 
        max_iter=1,  #Обучаем по одной итерации за раз
        warm_start=True,  #Сохраняем веса между итерациями
        tol=1e-8
    )
    
    #Постепенное обучение с сохранением истории потерь
    for i in range(max_iter):
        model.fit(xTrain, yTrain)
        
        yTrain_proba = model.predict_proba(xTrain)
        yTest_proba = model.predict_proba(xTest)
        
        trainLoss = log_loss(yTrain, yTrain_proba)
        testLoss = log_loss(yTest, yTest_proba)
        trainLosses.append(trainLoss)
        
        testLosses.append(testLoss)

        if i % 10 == 0:
            print(f"Iteration {i}: Train Loss = {trainLoss:.4f}, Test Loss = {testLoss:.4f}")
    
    return trainLosses, testLosses, model

#Без регуляризации, C большое
trainLoss_noReg, testLoss_noReg, model_noReg = trainLogisticRegressionWithTracking(
    c=1000000, 
    xTrain=xTrain_1, 
    xTest=xTest_1, 
    yTrain=yTrain, 
    yTest=yTest,
    max_iter=100
)

#Разумная регуляризация
trainLoss_withReg, testLoss_withReg, model_withReg = trainLogisticRegressionWithTracking(
    c=0.1, 
    xTrain=xTrain_1, 
    xTest=xTest_1, 
    yTrain=yTrain, 
    yTest=yTest,
    max_iter=100
)

#Сильная регуляризация
trainLoss_strongReg, testLoss_strongReg, model_strongReg = trainLogisticRegressionWithTracking(
    c=0.0001, 
    xTrain=xTrain_1, 
    xTest=xTest_1, 
    yTrain=yTrain, 
    yTest=yTest,
    max_iter=100
)

#Графики
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(trainLoss_noReg, label='Обучающая выборка', linewidth=2)
plt.plot(testLoss_noReg, label='Тестовая выборка', linewidth=2)
plt.title('Без регуляризации', fontsize=12, fontweight='bold')
plt.xlabel('Итерация')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 0.7)  #Фиксируем масштаб для сравнения

plt.subplot(2, 2, 2)
plt.plot(trainLoss_withReg, label='Обучающая выборка', linewidth=2)
plt.plot(testLoss_withReg, label='Тестовая выборка', linewidth=2)
plt.title('Разумная регуляризация', fontsize=12, fontweight='bold')
plt.xlabel('Итерация')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 0.7)

plt.subplot(2, 2, 3)
plt.plot(trainLoss_strongReg, label='Обучающая выборка', linewidth=2)
plt.plot(testLoss_strongReg, label='Тестовая выборка', linewidth=2)
plt.title('Сильная регуляризация', fontsize=12, fontweight='bold')
plt.xlabel('Итерация')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 0.7)

plt.subplot(2, 2, 4)
plt.plot(testLoss_noReg, label='Без регуляризации (C=10000)', linewidth=2)
plt.plot(testLoss_withReg, label='С регуляризацией (C=0.1)', linewidth=2)
plt.plot(testLoss_strongReg, label='Сильная регуляризация (C=0.001)', linewidth=2)
plt.title('Сравнение тестовых потерь', fontsize=12, fontweight='bold')
plt.xlabel('Итерация')
plt.ylabel('Log Loss (Test)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0.6, 0.7)

plt.tight_layout()
plt.show()

#Модель очень быстро сходится почти независимо от регуляризации. 
#Написал ли я криво или датасет такой - без понятия.