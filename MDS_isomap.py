import numpy as np
import pandas as df
from sklearn import metrics
from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义弗洛伊德算法函数，函数目的是搜索最近的K个邻近点的距离，以矩阵形式返回
def floyd(m,k=15):
	#取D矩阵中最大的值并乘以很大的数(这里藐视的是无穷大的意思，对应的Max就是无穷大)
	inf=np.max(m)*1000
	#获取D矩阵的维度
	n1,n2=m.shape
	#建立D1矩阵用于距离的计算
	d1=np.ones((n1,n1))*inf
	#把D按照行来排序返回排列的索引值
	m_arg=np.argsort(m,axis=1)
	#提取在每行中最小的K+1个数，其他的保持为d1矩阵原来的很大的值(这里类似于无穷大)
	for i in range(n1):
		d1[i,m_arg[i,0:k+1]]=m[i,m_arg[i,0:k+1]]
	#我们通过三次循环找到最小距离矩阵（详细步骤及原理请自行查看弗洛伊德算法原理）
	for a in range(n1):
		for i in range(n1):
			for j in range(n1):
				if d1[i,a]+d1[a,j]<d1[i,j]:
					d1[i,j]=d1[i,a]+d1[a,j]
	return d1
#按照老师上课所做的公式推导，计算高维数据的内积矩阵
def inner(m):
	(n1,n2)=m.shape
	d2=np.square(m)
	#按照行的方式把矩阵计算每行的数据的均方值
	di=np.sum(d2,axis=1)/n1
	#按照行的方式把矩阵计算每列的数据的均方值
	dj=np.sum(d2,axis=0)/n1
	#计算对应点数据在整个矩阵中的均方值
	dij=np.sum(d2)/(n1**2)
	#定义一个值为0的矩阵
	b=np.zeros((n1,n1))
	#对维数约简所需的内积矩阵来进行计算用b做容器返回矩阵
	for i in range(n1):
		for j in range(n2):
			b[i,j]=(dij+d2[i,j]-di[i]-dj[j])/(-2)
	return b
#定义MDS函数
def MDS(data,n=2):
	#先通过函数计算矩阵的距离
	m=metrics.pairwise.pairwise_distances(data,data)
	#计算距离的内积矩阵
	b=inner(m)
	#计算b矩阵的特征值fva和特征向量fve
	fva,fve=np.linalg.eigh(b)
	#对特征值做从大到小的排序，返回的是索引值
	fva_sort=np.argsort(-fva)
	#把特征值按照从大到小的顺序排列
	fva=fva[fva_sort]
	#把对应的特征向量也按照顺序排列
	fve=fve[:,fva_sort]
	#提取fva特征值的前n个值并使其对角化成矩阵（n表示要约简到的维度数）
	fvaz=np.diag(fva[0:n])
	#同样地提取对应的特征向量的前n个值
	fvez=fve[:,0:n]
	#按照老师黑板上说的计算，把约简后的特征值拆成两个特征值的平方根，然后再与对应的特征向量的转置做乘积，
	#最后所得的值便为降维后的点的值
	#这里方便计算，我还是直接不使用转置，直接计算特征向量与特征值的乘积就是新的点的值
	result=np.dot(fvez,np.sqrt(fvaz))
	return result
#定义ISOmap函数,这里的k表示搜索的邻域的点的个数，我们默认是30
def Isomap(data,n=2,k=30):
	m=metrics.pairwise.pairwise_distances(data,data)
	#这里在距离上我们采用的是弗洛伊德算法，这也是与MDS唯一不同的地方
	m=floyd(m,k)
	b=inner(m)
	fva,fve=np.linalg.eigh(b)
	fva_sort=np.argsort(-fva)
	fva=fva[fva_sort]
	fve=fve[:,fva_sort]
	fvaz=np.diag(fva[0:n])
	fvez=fve[:,0:n]
	result=np.dot(fvez,np.sqrt(fvaz))
	return result

if __name__=='__main__':
	#这里我们直接从sklearn所给出的样本中随机提取700个样本来进行维数约简的操作（data获取数据，color获取对应点的颜色）
	data,color=datasets.samples_generator.make_s_curve(300, random_state=9)
	#通过函数来对数据进行维数约简，这里使用的是ISOmap和MDS(默认采用的是降到二维)
	Isomap_data=Isomap(data)
	MDS_data=MDS(data)
	#定义画布和画布的大小
	figure=plt.figure(figsize=(15, 8))
	#设置大标题
	plt.suptitle('ISOMAP AND MDS COMPARE TO ORIGINAL DATA')
	#定义下面散点图的位置2*2的画布中第1个位置
	plt.subplot(2,2,1)
	#设置小标题
	plt.title('ISOMAP')
	#画出二维散点图
	plt.scatter(Isomap_data[:,0],Isomap_data[:,1],c=color,s=60)
	plt.subplot(2,2,3)
	plt.title('MDS')
	plt.scatter(MDS_data[:,0],MDS_data[:,1],c=color,s=60)
	#定义散点图为三维散点图
	ax =plt.subplot(2,2,2, projection='3d')
	plt.title('ORIGINAL DATA')
	ax.scatter(data[:,0],data[:,1],data[:,2], c=color, s=60)
	#显示图
	plt.show()
