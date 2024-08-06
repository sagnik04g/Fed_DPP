import torch
from torchvision import datasets, transforms
import json
from PIL import Image
import numpy as np
from inv_dpp import get_partitions
import pickle
from sklearn.preprocessing import LabelEncoder

# Define transformations
transform_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_2 = transforms.Compose([
    transforms.RandomRotation(30),  # Randomly rotate the image by up to 30 degrees
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# # Load the dataset

def save_imbalanced_data(split:str,s:int):
 if(split=='train'):
    train_split=True
 else:
    train_split=False
 
 le_cifar10 = LabelEncoder()


# # Load the dataset
 dataset1_cifar10 = datasets.CIFAR10(root='./data', train=train_split, download=True, transform=transform_1)
 dataset2_cifar10 = datasets.CIFAR10(root='./data', train=train_split, download=True, transform=transform_2)
 dataset1_mnist = datasets.MNIST(root='./data', train=train_split, download=True, transform=transform_1)
 dataset2_mnist = datasets.MNIST(root='./data', train=train_split, download=True, transform=transform_2)
 cifar10_classes=dataset1_cifar10.classes
 cifar10_train_labels1 = [cifar10_classes[label] for _, label in dataset1_cifar10]
 cifar10_train_labels2 = [cifar10_classes[label] for _, label in dataset2_cifar10]
 le_cifar10.fit(cifar10_classes)
 cifar10_dataset1_labels = le_cifar10.transform(cifar10_train_labels1)
 cifar10_dataset2_labels = le_cifar10.transform(cifar10_train_labels2)

 mnist_dataset1_labels = np.array(dataset1_mnist.train_labels)
 mnist_dataset2_labels = np.array(dataset2_mnist.train_labels)
 cifar10_data=np.concatenate([dataset1_cifar10.data.reshape(len(dataset1_cifar10.data),1024*3),dataset2_cifar10.data.reshape(len(dataset2_cifar10.data),1024*3)],axis=0)
 cifar10_labels=np.concatenate([cifar10_dataset1_labels,cifar10_dataset2_labels],axis=0)
 print('CIFAR10 data loaded. Generating imbalanced partitions...')
 cifar_dict={}
 all_client_data_dict_x_cifar10,all_client_data_dict_y_cifar10 = get_partitions(cifar10_data,cifar10_labels, 100, s, 5)
 for user_i in range(100):
   user_str=f'user_{user_i}'
   cifar_dict[user_str]={'x':all_client_data_dict_x_cifar10[user_i],'y':all_client_data_dict_y_cifar10[user_i]}
 cifar10_save_file=f'/scratch/sagnikg.scee.iitmandi/fl_dpp/data/imbalanced/{split}/cifar10_data_{split}_invdpp_s={s}.pickle'
 with open(cifar10_save_file, 'wb') as f:
    pickle.dump(cifar_dict,f)
    print('CIFAR10 imbalanced partitions generated')

 mnist_data=np.concatenate([dataset1_mnist.data.reshape(len(dataset1_mnist.data),784),dataset2_mnist.data.reshape(len(dataset2_mnist.data),784)],axis=0)
 mnist_labels=np.concatenate([mnist_dataset1_labels,mnist_dataset2_labels],axis=0)
 print('MNIST data loaded. Generating imbalanced partitions...')
 mnist_dict={}
 all_client_data_dict_x_mnist,all_client_data_dict_y_mnist = get_partitions(mnist_data, mnist_labels, 100, s, 5)
 for user_i in range(100):
   user_str=f'user_{user_i}'
   mnist_dict[user_str]={'x':all_client_data_dict_x_mnist[user_i],'y':all_client_data_dict_y_mnist[user_i]}
 mnist_save_file=f'/scratch/sagnikg.scee.iitmandi/fl_dpp/data/imbalanced/{split}/mnist_data_{split}_invdpp_s={s}.pickle'  
 with open(mnist_save_file, 'wb') as f:
    pickle.dump(mnist_dict,f)
    print('MNIST imbalanced partitions saved')

 femnist_json_path=f'/home/sagnikg.scee.iitmandi/Projects/TurboSVM-FL/data/femnist/all_data_0_niid_05_keep_64_{split}_8.json'
 celeba_json_path=f'/scratch/sagnikg.scee.iitmandi/FL_datasets/leaf/data/celeba/data/{split}/all_data_0_0_keep_3_{split}_9.json'
 shakespeare_json_path=f'/scratch/sagnikg.scee.iitmandi/FL_datasets/leaf/data/shakespeare/data/{split}/all_data_0_0_keep_0_{split}_9.json'
 with open(femnist_json_path, 'r') as f:
    femnist_data = json.load(f)
 femnist_data_dict={}
 fem_images,fem_labels=[],[]
 for user in femnist_data['users']:
    fem_images.append(np.array(femnist_data['user_data'][user]['x']))
    fem_labels.append(np.array(femnist_data['user_data'][user]['y']))
 fem_x_train = np.concatenate(fem_images,axis=0)
 fem_y_train = np.concatenate(fem_labels,axis=0)
 print('femnist data loaded. Generating imbalanced partitions...')
 fem_all_client_data_dict_x,fem_all_client_data_dict_y = get_partitions(fem_x_train, fem_y_train, 100, s, 5)
 for user_i in range(100):
   femnist_data_dict[user]={'x':fem_all_client_data_dict_x[user_i],'y':fem_all_client_data_dict_y[user_i]}
 femnist_save_path=f'/scratch/sagnikg.scee.iitmandi/fl_dpp/data/imbalanced/{split}/femnist_data_{split}_invdpp_s={s}.pickle'
 with open(femnist_save_path, 'wb') as f:
    pickle.dump(femnist_data_dict,f)
    print('femnist imbalanced partitions saved')

 with open(celeba_json_path, 'r') as f:
    celeba_data = json.load(f)
 image_path = '/scratch/sagnikg.scee.iitmandi/FL_datasets/leaf/data/celeba/data/raw/img_align_celeba/'
 celeba_images,celeba_labels=[],[]
 usr_images_dict={}
 for user in celeba_data['users']:
    for x in celeba_data['user_data'][user]['x']:
            img = Image.open(image_path + x)
            x_img= img.resize((84, 84)).convert('RGB')
            x_img = np.array(x_img).flatten()
            celeba_images.append(x_img)
    img_labels=np.array(celeba_data['user_data'][user]['y'],dtype=np.int64)
    celeba_labels.append(img_labels)

 celeba_x_train = np.array(celeba_images)
 celeba_y_train = np.concatenate(celeba_labels,axis=0)
 print('celeba data loaded. Generating imbalanced partitions...')
 all_client_data_dict_x_celeba,all_client_data_dict_y_celeba = get_partitions(celeba_x_train, celeba_y_train, 100, s, 5)
 for user_i in range(100):
   usr_images_dict[user]={'x':all_client_data_dict_x_celeba[user_i],'y':all_client_data_dict_y_celeba[user_i]}
 celeba_save_path=f'/scratch/sagnikg.scee.iitmandi/fl_dpp/data/imbalanced/{split}/celeba_data_{split}_invdpp_s={s}.pickle'
 with open(celeba_save_path, 'wb') as f:
    pickle.dump(usr_images_dict,f)
    print('celeba imbalanced partitions saved')

 with open(shakespeare_json_path, 'r') as f:
    shakespeare_data = json.load(f)
 skps_data_dict={}
 all_chars_sorted = ''' !"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz{}'''
 xs_skps, ys_skps = [], []
 for user in shakespeare_data['users']:
        # discard a user if it has too few samples
        for x, y in zip(shakespeare_data['user_data'][user]['x'], shakespeare_data['user_data'][user]['y']):
            assert(len(x) == 80)
            y = all_chars_sorted.find(y)
            if y == -1: # cannot find character
                raise Exception('wrong character:', y)
            ys_skps.append(y)
            
            x_pos_arr=np.array([all_chars_sorted.find(c) for c in x])
            # seq = torch.as_tensor(x_pos_arr)
            xs_skps.append(x_pos_arr)
            
 x_train_skps = np.array(xs_skps)
 print(f'Shape of a single sample is {x_train_skps[0].shape} ,and total samples is {len(xs_skps)} labels is {len(ys_skps)}')
 y_train_skps = np.array(ys_skps)
 print('shakespeare data loaded. Generating imbalanced partitions...')
 all_client_data_dict_x_skps,all_client_data_dict_y_skps = get_partitions(x_train_skps, y_train_skps, 100, s, 5)
 for user_i in range(100):
   skps_data_dict[user]={'x':all_client_data_dict_x_skps[user_i],'y':all_client_data_dict_y_skps[user_i]}
 shakespeare_save_path=f'/scratch/sagnikg.scee.iitmandi/fl_dpp/data/imbalanced/{split}/shakespearer_data_{split}_invdpp_s={s}.pickle'
 with open(shakespeare_save_path, 'wb') as f:
    pickle.dump(skps_data_dict,f)
    print('shakespeare imbalanced partitions saved')

save_imbalanced_data('train',5)
save_imbalanced_data('train',10)
save_imbalanced_data('train',15)
save_imbalanced_data('train',20)
save_imbalanced_data('train',25)
save_imbalanced_data('test',5)
save_imbalanced_data('test',10)
save_imbalanced_data('test',15)
save_imbalanced_data('test',20)
save_imbalanced_data('test',25)