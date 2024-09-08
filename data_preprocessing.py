import os
import torch
import json
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def get_data_dict_mnist(json_path: str, min_sample: int = 64, image_size: int = 28) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read MNIST data pickle file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """
    t=transforms.Compose(
        [transforms.Pad(18),
         transforms.Resize((64, 64)), transforms.ToTensor()])

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = pickle.load(f)
    final_data_dict={}
    for user,data in tmp_data_dict.items():
        if len(data['y']) < min_sample:
         continue

        ys_final = data['y']
        xs=[]
        for x in data['x']:
          x_img=np.array(x).reshape(image_size,image_size)
          x_img=Image.fromarray(np.uint8(x_img))
          x=t(x_img)
          xs.append(x)
        xs_final=torch.stack(xs).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_femnist(json_path: str, min_sample: int = 64, image_size: int = 28) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read MNIST data pickle file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)

    # if json_path.endswith(".json"):
    #  with open(json_path, 'r') as f:
    #     tmp_data_dict = json.load(f)
    # else:
    with open(json_path, 'rb') as f:
        tmp_data_dict = pickle.load(f)

    final_data_dict={}
    for user,data in tmp_data_dict.items():
        if len(data['y']) < min_sample:
         continue

        ys_final = data['y']
        xs_final=torch.as_tensor(data['x']).reshape(len(data['y']),1,image_size,image_size).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_cifar10(json_path: str, min_sample: int = 64, image_size: int = 32) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read CIFAR10 data json file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """
    t=transforms.Compose(
        [
         # transforms.Pad(16),  # Add padding on all sides
         # transforms.Resize((64, 64)),
         transforms.ToTensor()])

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = pickle.load(f)
    final_data_dict={}
    for user,data in tmp_data_dict.items():
        if len(data['y']) < min_sample:
          continue

        ys_final = data['y']
        xs=[]
        for x in data['x']:
          x_img=np.array(x).reshape(image_size,image_size,3)
          x_img=Image.fromarray(np.uint8(x_img))
          x=t(x_img)
          xs.append(x)
        xs_final=torch.stack(xs).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_shakespeare(json_path: str, min_sample: int = 64, seq_len: int = 80) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read FEMNIST data json file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = pickle.load(f)
    final_data_dict={}
    for user,data in tmp_data_dict.items():
        if len(data['y']) < min_sample:
         continue

        xs_final = []
        for x in data['x']:
            assert(len(x) == seq_len)
            x = torch.as_tensor(x)
            xs_final.append(x)

        ys_final = data['y']
        xs_final=torch.stack(xs_final)
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict

def get_data_dict_celeba(json_path: str, min_sample: int = 1, image_size: int =84) -> dict[str, dict[str, torch.Tensor]]:
    """
    Read FEMNIST data json file and save into dictionary.

    Arguments:
        json_path (str): path to data json file.
        min_sample (int): minimal number of samples per client.
        image_size (int): height / width of images. The images should be of rectangle shape.

    Returns:
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.
    """

    if not os.path.exists(json_path):
        raise Exception("file doesnt exist:", json_path)
    
    with open(json_path, 'rb') as f:
        tmp_data_dict = pickle.load(f)
    final_data_dict={}
    for user,data in tmp_data_dict.items():
        if len(data['y']) < min_sample:
         continue

        xs_final = []
        for x in data['x']:
            x = torch.as_tensor(x).reshape(3, image_size, image_size)
            xs_final.append(x)
        ys_final = data['y']

        xs_final=torch.stack(xs_final).float()
        ys_final=torch.as_tensor(ys_final).long()
        final_data_dict[user]={'x' : xs_final,'y' : ys_final}

    return final_data_dict
class Dataset(torch.utils.data.Dataset):
    """
    Self-defined dataset class.
    """

    def __init__(self, xs: torch.Tensor, ys: torch.Tensor) -> None:
        """
        Arguments:
            xs (torch.Tensor): samples.
            ys (torch.Tensor): ground truth labels.
        """

        self.xs = xs
        self.ys = ys
        
    def __len__(self) -> int:
        """
        Returns:
            (int): size of dataset.
        """

        return len(self.ys)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            idx (int): index to sample.

        Returns:
            x (torch.Tensor): sample.
            y (torch.Tensor): ground truth label.
        """

        x = self.xs[idx]
        y = self.ys[idx]

        return x, y