from sklearn.cluster import KMeans
import numpy as np
import time
import torch

def get_partitions(x_train,y_train, P, P_select,k_clusters):

    x_train = np.reshape(x_train, [x_train.shape[0], -1])
    y_train = np.reshape(y_train, [y_train.shape[0], ])

    classes = np.unique(y_train)
    print(f'Total no. of classes is {len(classes)} and total classes are {classes}')
    c_data = []
    party = np.zeros(y_train.shape)
    for i,c in enumerate(classes):
        class_idx = np.where(y_train == c)[0]
        c_data.append(x_train[class_idx])
        print(c_data[i].shape)
        if (len(c_data[i]) < k_clusters):
         continue
        start_time = time.time()
        kmeanst = KMeans(n_clusters = k_clusters, init='k-means++', max_iter = 10, random_state = 42)
        kmeanst.fit(c_data[i])
        end_time = time.time()
        print('Kmeans execution time in seconds: {}'.format(end_time - start_time))
        for k in np.unique(kmeanst.labels_):
            print(k)
            cluster_idx = list(np.where(kmeanst.labels_ == k)[0])
            P_selected = np.random.choice(range(P),P_select,replace=False)
            P_ratio = np.random.rand(P_select)
            P_ratio=P_ratio/np.sum(P_ratio)
            P_ratio=np.round(len(cluster_idx)*P_ratio)
            start_idx=0
            end_idx=0
            for p in range(P_select):
                if P_ratio[p]+start_idx>len(cluster_idx):
                    end_idx=len(cluster_idx)
                else:
                    end_idx=int(P_ratio[p]+start_idx)
                party[list(class_idx[cluster_idx[start_idx:end_idx]])]=P_selected[p]
                start_idx=end_idx
    x_trains=[]
    y_trains=[]
    for p in range(P):
        x_trains.append(x_train[party == p])
        y_trains.append(y_train[party == p])
        print(x_trains[p].shape)
        print(y_trains[p].shape)
    # xs=[]
    # ys=[]
    # for xs_i in x_trains:
    #     xs.append(torch.as_tensor(xs_i).reshape(len(xs_i),1,28,28))
    # for ys_i in y_trains:
    #     ys.append(torch.as_tensor(ys_i).long())
    return x_trains,y_trains
