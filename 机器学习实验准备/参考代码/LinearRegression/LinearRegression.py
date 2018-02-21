#encoding=utf-8
import matplotlib.pyplot as plt   #画图工具
import numpy as np
import pandas as pd    #使用pandas读取csv数据
from sklearn import datasets,linear_model
#from sklearn.cross_validation import train_test_split   #适用于0.18之前的版本
from sklearn.model_selection import train_test_split    #适用于0.20及之后的版本
from sklearn.linear_model import LinearRegression
from sklearn import metrics  # 引入sklearn模型评价工具库

"""
读取样本数据，并将数据集分为训练集和测试集
"""
def getTrainSetAndTestSet(DataPath):
    data = pd.read_csv(DataPath)
    X = data[['AT','V','AP','RH']]  #AT， V，AP和RH这4个列作为样本特征。
    y = data[['PE']]                 #用PE作为样本输出
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)  #随机划分训练集和测试集，默认把数据集的25%作为测试集
    #查看训练集和测试集的维度
    print "训练集和测试集的维度:"
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape
    return X_train,X_test,y_train,y_test

"""
训练Linear Regreesion模型，得到训练参数
"""
def TrainLinearRegression(X_train,y_train):
    linreg = LinearRegression()   #未经训练的机器学习模型
    linreg.fit(X_train,y_train)  #对模型传入输入数据x_train和输出数据y_train并进行训练
    """
    输出线性回归的截距和各个系数,得到线性回归预测的回归函数：
    PE=447.06297099−1.97376045∗AT−0.23229086∗V+0.0693515∗AP−0.15806957∗RH
    """
    print "线性回归各个系数(W)：", linreg.coef_
    print "线性回归参数截距(b)：",linreg.intercept_
    return linreg

"""
使用均方误差（Mean Squared Error, MSE）和均方根误差(Root Mean Squared Error, RMSE)在测试集上的表现来评价模型的好坏。
"""
def EvaluationModel(linreg,X_test,y_test):
    y_pred = linreg.predict(X_test)
    # 用scikit-learn计算MSE
    print "均方误差MSE:",metrics.mean_squared_error(y_test,y_pred)
    #用scikit-learn计算RMSE
    print "均方根误差RMSE:",np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    return y_pred

"""
可视化的方式直观的表示模型学习效果的好坏
对于输出y来说，真实值和预测值都是一维的，同时，真实值和预测值一一对应，它们之间的差值越小，预测越准确。
显然，如果预测值=真实值，那么它们的差值最小，即上图中的黑色虚线。横坐标是真实值，纵坐标是预测值，那么对于所有的真实值，
预测值离着黑线越近，预测越准确。
"""
def Visualization(y_test,y_pred):
    fig,ax = plt.subplots()
    ax.scatter(y_test,y_pred)
    ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'k--',lw=5)  # ’k–-’,k指线为黑色，–是线的形状。lw指定线宽。
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()

if __name__ == "__main__":
    DataPath = "./Data/Folds5x2_pp.csv"    #数据的相对路径
    X_train, X_test, y_train, y_test = getTrainSetAndTestSet(DataPath)
    linreg = TrainLinearRegression(X_train,y_train)
    y_pred = EvaluationModel(linreg, X_test, y_test)
    Visualization(y_test, y_pred)
