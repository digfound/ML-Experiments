#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
--------获取数据--------
随机生成40个独立的点，一共有两类
X数据格式[[6.37734541 ,-10.61510727],[6.50072722 ,-3.82403586]...]
Y数据格式[1 0 1 0 1 ...1 0 1]   0和1代表类别
"""
X,y = make_blobs(n_samples=40,centers=2,random_state=6)   #随机生成2类数据，一共含40个样本
"""
print X
print y
"""

"""
--------训练SVM模型--------
kernel='linear' 核函数选择Linear核，主要用于线性可分的情形。参数少，速度快。
C=1000 惩罚参数，C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，
趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，
将他们当成噪声点，泛化能力较强。
"""
clf = svm.SVC(kernel='linear',C=1000)    #SVC时SVM的一种Type,是用来做分类的
clf.fit(X,y)      #喂给模型数据，进行训练
print "模型参数W:",clf.coef_    #模型参数W
print "模型参数b:",clf.intercept_     #模型参数b
print "支持向量为：",clf.support_vectors_   #输出支持向量
xpredict = [10,-10]
xpredict = np.array(xpredict).reshape(1,-1)   #sklearn 0.17版本之后需要这条语句，之前版本直接传入xpredict即可
print xpredict,"预测为:",clf.predict(xpredict),"类别"
xpredict = [10,-2]
xpredict = np.array(xpredict).reshape(1,-1)
print xpredict,"预测为:",clf.predict(xpredict),"类别"

"""
--------可视化(作为了解)--------
"""

plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired) #X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
#plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()       #获取X轴的范围
ylim = ax.get_ylim()       #获取Y轴的范围
#create grid to evaluate model
xx = np.linspace(xlim[0],xlim[1],30)   # 返回30个均匀分布的样本，在[xlim[0],xlim[1]]范围内
yy = np.linspace(ylim[0],ylim[1],30)
YY,XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
#绘出支持向量的分割界和分割面（二维分割面为直线）
ax.contour(XX,YY,Z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
#绘出支持向量，用红色表示
ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],linewidth=1,facecolors='red')
plt.show()



