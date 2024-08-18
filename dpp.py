import os
import torch
import numpy as np
import logging
from PIL import Image
from sklearn.mixture import GaussianMixture
import torchvision.transforms as transforms
import pickle

def get_balanced_partitions(file_path:str,split:str='train',dataset:str='mnist',min_sample:int=8,shape: tuple = (1,28,28)):
   with open(file_path,'rb') as f:
    data_dict=pickle.load(f)
   all_xs,all_ys=[],[]
   
   for user,data in data_dict.items():
     data_x=data['x']
     all_xs.append(data_x)
     data_y=data['y']
     all_ys.append(data_y)
   x_train = np.concatenate(all_xs,axis=0)
   y_train = np.concatenate(all_ys,axis=0)
   classes=np.unique(y_train)
   print(x_train.shape,y_train.shape)
   logging.debug(f'{x_train.shape},{y_train.shape}')
   X_class_wise,Y_class_wise=[],[]
   for i in classes: 
     if(len(x_train[y_train==i])>0):
      X_class_wise.append(x_train[y_train==i])
      Y_class_wise.append(y_train[y_train==i])
   means_clss_wise,sigma_clss_wise,weights_clss_wise,data_points_clss_wise=get_gmm_clusters(X_class_wise,Y_class_wise,5)
   for user,data in data_dict.items():
        if len(data['y']) < min_sample:
         continue

        # xs_flattened,ys_flattened=np.array(data['user_data'][user]['x']),np.array(data['user_data'][user]['y'])
        xs_final = []
        for x in data['x']:
            if(dataset=='shakespeare'):
               x=torch.as_tensor(x)
            else:
               x = torch.as_tensor(x).reshape(shape)
            xs_final.append(x)
        ys_final = data['y']

        xs_flattened,ys_flattened=np.array(data['x']),np.array(data['y'])

        print(xs_flattened.shape,ys_flattened.shape)
        logging.debug(f'{xs_flattened.shape},{ys_flattened.shape}')
        for i in range(classes):
         client_x=xs_flattened[ys_flattened==i]
         clusters_all=data_points_clss_wise[i]
         all_data_points_clss=np.sum([len(clusters_all[j]) for j in range(len(data_points_clss_wise[i]))])
         ratio_clusters_all=[(len(clusters_all[j])/all_data_points_clss) for j in range(len(data_points_clss_wise[i]))]
         print(f"Class {i},Shape={client_x.shape}")
         #logging.debug(f'Class {i},Shape={client_x.shape}')
         
         if(client_x.shape[0] < 10):
           difference_sample=np.ceil(np.array(ratio_clusters_all)*10)
           difference_sample=[int(x) for x in difference_sample]
        
         else:
          clusters=[[] for _ in data_points_clss_wise[i]]
          for clust_i,clust in enumerate(data_points_clss_wise[i]):
             for data_point in client_x:
                 if(data_point in clust):
                    clusters[clust_i].append(data_point)


          ratio_clusters_client=[(len(clusters[j])/client_x.shape[0]) for j in range(len(clusters))]
          difference_ratio=np.array([abs(a-b) for a,b in zip(ratio_clusters_all,ratio_clusters_client)])
          del clusters
          difference_sample=np.ceil(difference_ratio*client_x.shape[0])
          difference_sample=[int(x) for x in difference_sample]
         for k in range(5):
            if(difference_sample[k] != 0):
             print("adjusting distribution for each client")
             logging.debug('adjusting distribution for each client')
             extra_sample=np.random.multivariate_normal(mean=means_clss_wise[i][k], cov=sigma_clss_wise[i][k], size=difference_sample[k]).astype(np.float32)
             for sample in extra_sample:
                samples_tensor=torch.as_tensor(sample).reshape(shape).float()
                xs_final.append(samples_tensor)
             extra_ys=np.full((difference_sample[k]),i)
             ys_final=np.concatenate([ys_final,extra_ys])
             print(f"added {difference_sample[k]} samples in user {user}")
             logging.debug(f"added {difference_sample[k]} samples in user {user}")

         del client_x
         del clusters_all
         del difference_sample

        xs_final=torch.stack(xs_final)
        ys_final=torch.as_tensor(ys_final).long()
        data_dict[user]={'x' : xs_final,'y' : ys_final}
        logging.debug(f'User {user} shape: x is {xs_final.shape}, y is {ys_final.shape}')
        print(f'User {user} shape: x is {xs_final.shape}, y is {ys_final.shape}')

   with open(f'/scratch/sagnikg.scee.iitmandi/fl_dpp/data/balanced/{split}/{dataset}_balanced_100_clients.json', 'wb') as f:
     pickle.dump(data_dict, f)

def get_gmm_clusters(X_class_wise,Y_class_wise,k_clusters):
    cluster_means_class_wise,cluster_sigma_class_wise,cluster_weights_class_wise,clusters_class_wise=[],[],[],[]
    for x,y in zip(X_class_wise,Y_class_wise):
      clusters=[]
      gmm = GaussianMixture(n_components=k_clusters,init_params='kmeans', covariance_type='full',reg_covar=1e-3,max_iter=30)
      if(len(x)>=k_clusters):
        gmm.fit(x)
        labels=gmm.predict(x)
        for cnt in range(k_clusters):
          clusters.append(x[labels==cnt])
        means=gmm.means_
        covariances=gmm.covariances_
        weights=gmm.weights_
      else:
        for cnt in range(k_clusters):
          clusters.append(x)
        means=np.zeros((5,len(x[0])))
        covariances=np.ones((5,int(len(x[0])),len(x[0])))
        weights=np.ones((5, len(x[0])))
      cluster_means_class_wise.append(means)
      cluster_sigma_class_wise.append(covariances)
      cluster_weights_class_wise.append(weights)
      clusters_class_wise.append(clusters)
    return cluster_means_class_wise,cluster_sigma_class_wise,cluster_weights_class_wise,clusters_class_wise

get_balanced_partitions('..data/imbalanced/train/mnist_data_train_iid.json','train','mnist',64,(1,28,28))
get_balanced_partitions('..data/imbalanced/test/mnist_data_test_iid.json','test','mnist',64,(1,28,28))
get_balanced_partitions('..data/imbalanced/train/femnist_data_train_iid.json','train','femnist',64,(1,28,28))
get_balanced_partitions('..data/imbalanced/test/femnist_data_test_iid.json','test','femnist',64,(1,28,28))
get_balanced_partitions('..data/imbalanced/train/cifar10_data_train_iid.json','train','cifar10',8,(3,32,32))
get_balanced_partitions('..data/imbalanced/test/cifar10_data_test_iid.json','test','cifar10',8,(3,32,32))
get_balanced_partitions('..data/imbalanced/train/celeba_data_train_iid.json','train','celeba',8,(3,84,84))
get_balanced_partitions('..data/imbalanced/test/celeba_data_test_iid.json','test','celeba',8,(3,84,84))
get_balanced_partitions('..data/imbalanced/train/shakespeare_data_train_iid.json','train','shakespeare',8)
get_balanced_partitions('..data/imbalanced/train/shakespeare_data_test_iid.json','test','shakespeare',8)
get_balanced_partitions('..data/imbalanced/test/celeba_data_test_iid.json','test','celeba',8,(3,84,84))
