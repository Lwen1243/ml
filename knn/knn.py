import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from sklearn.externals import joblib

# 数据预处理
trainData = np.loadtxt(open('digits_training.csv', 'r'), delimiter=",", skiprows=1)  # 装载数据
MTrain, NTrain = np.shape(trainData)  # 行列数
print("训练集：", MTrain, NTrain)
xTrain = trainData[:, 1:NTrain]
xTrain_col_avg = np.mean(xTrain, axis=0)  # 对各列求均值
xTrain = (xTrain - xTrain_col_avg) / 255  # 归一化
yTrain = trainData[:, 0]

'''================================='''
# 训练模型
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500)
model.fit(xTrain, yTrain)
print("训练完毕")

'''================================='''
# 测试模型
testData = np.loadtxt(open('digits_testing.csv', 'r'), delimiter=",", skiprows=1)
MTest, NTest = np.shape(testData)
print("测试集：", MTest, NTest)
xTest = testData[:, 1:NTest]
xTest = (xTest - xTrain_col_avg) / 255  # 使用训练数据的列均值进行处理
yTest = testData[:, 0]
yPredict = model.predict(xTest)
errors = np.count_nonzero(yTest - yPredict)  # 返回非零项个数
print("预测完毕。错误：", errors, "条")
print("测试数据正确率:", (MTest - errors) / MTest)

'''================================='''
# 保存模型

# 创建文件目录
dirs = 'testModel'
if not os.path.exists(dirs):
    os.makedirs(dirs)
joblib.dump(model, dirs + '/model.pkl')
print("模型已保存")
