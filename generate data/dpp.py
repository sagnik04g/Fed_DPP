import os
import numpy as np
import json
import logging
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.mixture import GaussianMixture
import pickle
import argparse
from sklearn.decomposition import PCA


def Args() -> argparse.Namespace:
    """
    Helper function for argument parsing.

    Returns:
        args (argparse.Namespace): parsed argument object.
    """

    parser = argparse.ArgumentParser()
    
    # path parameters
    parser.add_argument('--file_path', type = str, default = '../data/imbalanced/test/celeba_data_test_invdpp_s=50.pickle', help = 'celeba train json path')
    parser.add_argument('--split' , type = str, default = 'train'  , help = 'dataset split')
    parser.add_argument('--dataset' , type = str, default = 'celeba'  , help = 'dataset name')
    parser.add_argument('--min_sample' , type = int, default = '64'  , help = 'dataset')
    parser.add_argument('--shape' , type = tuple, default = (1,84,84,3)  , help = 'data x shape')
    parser.add_argument('--k_clusters' , type = int, default = '10'  , help = 'gmm components')
        
    args = parser.parse_args()

    return args


def get_balanced_partitions(file_path:str,split:str='train',dataset:str='mnist',min_sample:int=8,shape: tuple = (1,28,28), k_clusters: int=5):
   all_xs,all_ys=[],[]
   if(file_path.endswith('.pickle')):
    with open(file_path,'rb') as f:
     data_dict=pickle.load(f)
     for user,data in data_dict.items():
      data_x=data['x']
      all_xs.append(np.array(data_x))
      data_y=data['y']
      all_ys.append(np.array(data_y))
   
   if(file_path.endswith('.json')):
    with open(file_path,'r') as f:
     temp_data_dict=json.load(f)
    data_dict={}
    for user, num_sample in zip(temp_data_dict['users'], temp_data_dict['num_samples']):
        xs=[]
        for x in temp_data_dict['user_data'][user]['x']:
            xs.append(np.array(x))
            all_xs.append(np.array(x))
        ys = np.array(data['user_data'][user]['y'])
        all_ys.append(ys)
        data_dict[user] = {'x' : xs, 'y' : ys}
   
   x_train = np.concatenate(all_xs,axis=0)
   x_train = x_train.reshape(len(x_train),shape[1],shape[2])
   y_train = np.concatenate(all_ys,axis=0)
   classes=np.unique(y_train)
   print(x_train.shape,y_train.shape)
   logger.debug(f'{x_train.shape},{y_train.shape}')
   X_class_wise,Y_class_wise=[],[]
   for i in classes:
      X_class_wise.append(x_train[y_train==i])
      Y_class_wise.append(y_train[y_train==i])
   means_clss_wise,sigma_clss_wise,weights_clss_wise,data_points_clss_wise=get_gmm_clusters(X_class_wise,Y_class_wise,k_clusters)
   for user,data in data_dict.items():
        if len(data['y']) < min_sample:
         continue

        # xs_flattened,ys_flattened=np.array(data['user_data'][user]['x']),np.array(data['user_data'][user]['y'])
        xs_final = []
        for x in data['x']:
            if dataset=='shakespeare':
              x=np.array(x,dtype=np.int64)
            else:
              x=np.array(x)
            x = x.reshape(shape)
            xs_final.append(x)
        ys_final = data['y']

        xs_flattened,ys_flattened=np.array(data['x']),np.array(data['y'])

        print(xs_flattened.shape,ys_flattened.shape)
        logger.debug(f'{xs_flattened.shape},{ys_flattened.shape}')
        for i,clss in enumerate(classes):
         client_x=xs_flattened[ys_flattened==clss]
         clusters_all=data_points_clss_wise[i]
         all_data_points_clss=np.sum([len(clusters_all[j]) for j in range(len(data_points_clss_wise[i]))])
         ratio_clusters_all=[(len(clusters_all[j])/all_data_points_clss) for j in range(len(data_points_clss_wise[i]))]
         print(f"Class {clss},Shape={client_x.shape}")
         #logging.debug(f'Class {i},Shape={client_x.shape}')
         
         if(client_x.shape[0] < 10):
           difference_sample=np.ceil(np.array(ratio_clusters_all)*10)
           difference_sample=[int(x) for x in difference_sample]
        
         else:
          # StreamKmeans- 1 iter clusters
          kmeans_x = KMeans(n_clusters = k_clusters, init=means_clss_wise[i], max_iter = 1, random_state = 42)
          kmeans_x.fit(client_x)
          clusters = [client_x[kmeans_x.labels_== i] for i in range(k_clusters)]

          # Naive-approach for Stream Kmeans - Alternate method
          # clusters=[[]] for _ in k_clusters]
          # for clust_i,clust in enumerate(data_points_clss_wise[i]):
          #    for data_point in client_x:
          #        if(data_point in clust):
          #           clusters[clust_i].append(data_point)


          ratio_clusters_client=[(len(clusters[j])/client_x.shape[0]) for j in range(len(clusters))]
          difference_ratio=np.array([(a-b) if (a-b)>0 else 0 for a,b in zip(ratio_clusters_all,ratio_clusters_client)])
          del clusters
          difference_sample=np.ceil(difference_ratio*client_x.shape[0])
          difference_sample=[int(x) for x in difference_sample]
         for k in range(k_clusters):
            if(difference_sample[k] != 0):
             print("adjusting distribution for each client")
             logger.debug('adjusting distribution for each client')
             extra_sample=np.random.multivariate_normal(mean=means_clss_wise[i][k], cov=sigma_clss_wise[i][k], size=difference_sample[k]).astype(np.float32)
             for sample in extra_sample:
                if(dataset=='shakespeare'):
                 sample_list=[c if c<=79 or c>=0 else 0 for c in sample]
                 samples_tensor=np.array(sample_list,dtype=np.int64).reshape(shape)
                else:
                 samples_tensor=np.array(sample).reshape(shape)
                xs_final.append(samples_tensor)
             extra_ys=np.full((difference_sample[k]),clss)
             ys_final=np.concatenate([ys_final,extra_ys])
             print(f"added {difference_sample[k]} samples in user {user}")
             logger.debug(f"added {difference_sample[k]} samples in user {user}")

         del client_x
         del clusters_all
         del difference_sample

        xs_final=np.concatenate(xs_final)
        ys_final=np.array(ys_final)
        data_dict[user]={'x' : xs_final,'y' : ys_final}
        logger.debug(f'User {user} shape: x is {xs_final.shape}, y is {ys_final.shape}')
        print(f'User {user} shape: x is {xs_final.shape}, y is {ys_final.shape}')

   with open(f'../data/balanced/{split}/{dataset}_balanced_100_clients.pickle', 'wb') as f:
     pickle.dump(data_dict, f)

def get_gmm_clusters(X_class_wise,Y_class_wise,k_clusters):
    cluster_means_class_wise,cluster_sigma_class_wise,cluster_weights_class_wise,clusters_class_wise=[],[],[],[]
    for x,y in zip(X_class_wise,Y_class_wise):
      clusters=[]
      gmm = GaussianMixture(n_components=k_clusters,init_params='k-means++', covariance_type='full',reg_covar=1e-3,max_iter=10)
      if(len(x)<k_clusters):
        x=np.concatenate([x]*k_clusters)
      gmm.fit(x)
      for cnt in range(k_clusters):
        clusters.append(x[gmm.labels==cnt])
      means=gmm.means_
      covariances=gmm.covariances_
      weights=gmm.weights_
      cluster_means_class_wise.append(means)
      cluster_sigma_class_wise.append(covariances)
      cluster_weights_class_wise.append(weights)
      clusters_class_wise.append(clusters)
    return cluster_means_class_wise,cluster_sigma_class_wise,cluster_weights_class_wise,clusters_class_wise


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    # Set the logging level
    logger.setLevel(logging.DEBUG)


    # Create a file handler
    file_handler = logging.FileHandler('dpp_log.log')

    # Set the formatting for the file handler
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    args = Args()
    get_balanced_partitions(args.file_path,args.split,args.dataset,args.min_sample,args.shape,args.k_clusters)
