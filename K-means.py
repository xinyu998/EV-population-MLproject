#data=EV_popu_WA_data
#X=EV_popu_WA_data[['longitudinal','lateral']]

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

data=pd.read_excel('https://github.com/xinyu998/WA-EV-population-ML-project/blob/main/EVprocessed_data.xlsx?raw=true')
X=data[['longitudinal','lateral']]

## K-mean Elbow method to find the best k value
from sklearn import metrics
from scipy.spatial.distance import cdist
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K=range(2,12)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
 
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_
for key, val in mapping1.items():
    print(f'{key} : {val}')
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()

kmeans_model=KMeans(init="k-means++",n_clusters=6)
kmeans_model.fit(X)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X.iloc[:, 0].min()-1, X.iloc[:, 0].max()+1
y_min, y_max = X.iloc[:, 1].min()-1, X.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X.iloc[:, 0], X.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids = kmeans_model.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()

# get sample points from each cluster
#data=pd.read_excel('https://github.com/xinyu998/WA-EV-population-ML-project/blob/main/EVprocessed_data.xlsx?raw=true')

"""**Plot of Each Clusters**"""

#i th cluster 

## get sample points from each cluster
cluster_map=pd.DataFrame()
cluster_map['data_index'] = data.index.values
cluster_map['cluster'] = kmeans_model.labels_
cluster_map['longitudinal']=data.longitudinal
cluster_map['lateral']=data.lateral
cluster_map['year']=data.ModelYear
cluster_map['make']=data.Make
cluster_map['range']=data.ElectricRange
#cluster_map.to_excel('cluster3.xlsx')

# fig, axs = plt.subplots(2,3)
# for i,axs in zip(range (0,6),axs.ravel()):
#     cluster= cluster_map.loc[cluster_map['cluster']==i]
#     x_min, x_max = cluster.iloc[:,2].min()-0.2, cluster.iloc[:,2].max()+0.2
#     y_min, y_max = cluster.iloc[:,3].min()-0.2, cluster.iloc[:,3].max()+0.2
#     axs.plot(cluster.iloc[:,2], cluster.iloc[:, 3],'k.',markersize=2)
#     axs.set_xlim(x_min, x_max)
#     axs.set_ylim(y_min, y_max)
#     axs.set_title('cluster '+str([i]))
# plt.show()

"""**Shape of Clusters**"""

for i in range (0,6):
  cluster= cluster_map.loc[cluster_map['cluster']==i]
  print("cluster"+str(i) +"shape: ", cluster.shape)

cluster0=cluster_map.loc[cluster_map['cluster']==0]
X0=cluster0[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X0)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X0.iloc[:, 0].min()-1, X0.iloc[:, 0].max()+1
y_min, y_max = X0.iloc[:, 1].min()-1, X0.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X0.iloc[:, 0], X0.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids0 = kmeans_model.cluster_centers_
plt.scatter(centroids0[:, 0], centroids0[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
cluster0['newcluster']=kmeans_model.labels_
cluster0.to_excel('cluster0.xlsx')

#dividing cluster0
cluster0_0=cluster0.loc[cluster0['newcluster']==0]
X00=cluster0_0[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X00)
centroids00 = kmeans_model.cluster_centers_
cluster0_0['new2cluster']=kmeans_model.labels_
cluster0_0.to_excel('cluster00.xlsx')

cluster0_1=cluster0.loc[cluster0['newcluster']==1]
X01=cluster0_1[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X01)
centroids01 = kmeans_model.cluster_centers_
cluster0_1['new2cluster']=kmeans_model.labels_
cluster0_1.to_excel('cluster01.xlsx')

cluster0_2=cluster0.loc[cluster0['newcluster']==2]
X02=cluster0_2[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X02)
centroids02 = kmeans_model.cluster_centers_
cluster0_2['new2cluster']=kmeans_model.labels_
cluster0_2.to_excel('cluster02.xlsx')

cluster0_3=cluster0.loc[cluster0['newcluster']==3]
X03=cluster0_3[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X03)
centroids03 = kmeans_model.cluster_centers_
cluster0_3['new2cluster']=kmeans_model.labels_
cluster0_3.to_excel('cluster03.xlsx')

cluster0_4=cluster0.loc[cluster0['newcluster']==4]
X04=cluster0_4[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X04)
centroids04 = kmeans_model.cluster_centers_
cluster0_4['new2cluster']=kmeans_model.labels_
cluster0_4.to_excel('cluster04.xlsx')

#average distance of EV and Charging station
cluster0_0=cluster0.loc[cluster0['newcluster']==0]
#calculate number of charging ports
range00=np.mean(cluster0_0[['range']])
X00=cluster0_0[['longitudinal','lateral']]
size=X00.shape
print(size)
#dis_value=[]
#for i in range (0,9049):
 # dis_value[i,0]=((X00.iloc[i,0]*54.6-centroids0[0,0]*54.6)**2+(X00.iloc[i,1]*69-centroids0[0,1]*69)**2)**0.5
#print(dis_value)
#dis_mean=np.mean(dis_value) 
#print(dis_mean)

cluster1=cluster_map.loc[cluster_map['cluster']==1]
X1=cluster1[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X1)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X1.iloc[:, 0].min()-1, X1.iloc[:, 0].max()+1
y_min, y_max = X1.iloc[:, 1].min()-1, X1.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X1.iloc[:, 0], X1.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids1 = kmeans_model.cluster_centers_
plt.scatter(centroids1[:, 0], centroids1[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
cluster1['newcluster']=kmeans_model.labels_
cluster1.to_excel('cluster1.xlsx')




cluster2=cluster_map.loc[cluster_map['cluster']==2]
X2=cluster2[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X2)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X2.iloc[:, 0].min()-1, X2.iloc[:, 0].max()+1
y_min, y_max = X2.iloc[:, 1].min()-1, X2.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X2.iloc[:, 0], X2.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids2 = kmeans_model.cluster_centers_
plt.scatter(centroids2[:, 0], centroids2[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
cluster2['newcluster']=kmeans_model.labels_
cluster2.to_excel('cluster2.xlsx')




cluster3=cluster_map.loc[cluster_map['cluster']==3]
X3=cluster3[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=6)
kmeans_model.fit(X3)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X3.iloc[:, 0].min()-1, X3.iloc[:, 0].max()+1
y_min, y_max = X3.iloc[:, 1].min()-1, X3.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X3.iloc[:, 0], X3.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids3 = kmeans_model.cluster_centers_
plt.scatter(centroids3[:, 0], centroids3[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
cluster3['newcluster']=kmeans_model.labels_
cluster3.to_excel('cluster3.xlsx')



cluster4=cluster_map.loc[cluster_map['cluster']==4]
X4=cluster4[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X4)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X4.iloc[:, 0].min()-1, X4.iloc[:, 0].max()+1
y_min, y_max = X4.iloc[:, 1].min()-1, X4.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X4.iloc[:, 0], X4.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids4 = kmeans_model.cluster_centers_
plt.scatter(centroids4[:, 0], centroids4[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
cluster4['newcluster']=kmeans_model.labels_
cluster4.to_excel('cluster4.xlsx')




cluster5=cluster_map.loc[cluster_map['cluster']==5]
X5=cluster5[['longitudinal','lateral']]
kmeans_model=KMeans(init="k-means++",n_clusters=5)
kmeans_model.fit(X5)
h=.02
#print(X.iloc[:,0].min())
# Plot the decision boundary
x_min, x_max = X5.iloc[:, 0].min()-1, X5.iloc[:, 0].max()+1
y_min, y_max = X5.iloc[:, 1].min()-1, X5.iloc[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#Obtain labels for each point in mesh. Use last trained model.
Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15,8))
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='coolwarm', aspect="auto", origin="lower")

# Plot the projected data on 2D space
plt.plot(X5.iloc[:, 0], X5.iloc[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids5 = kmeans_model.cluster_centers_
plt.scatter(centroids5[:, 0], centroids5[:, 1], marker="x", s=120, linewidths=5, color="w", zorder=10)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.show()
cluster5['newcluster']=kmeans_model.labels_
cluster5.to_excel('cluster5.xlsx')


#Plot the total station area
SC=np.concatenate((centroids00,centroids01,centroids02,centroids03,centroids04,centroids1,centroids2,centroids3,centroids4,centroids5))
print(SC)
plt.figure()
plt.scatter(SC[:,0], SC[:,1], marker="^", s=120, linewidths=1, color="blue", zorder=10)
plt.show()
SC=pd.DataFrame(SC)
SC.to_excel('Chargingstation.xlsx')