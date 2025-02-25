"""
Module Name: data_loader.py
Author: Alice Bizeul
Ownership: ETH Zürich - ETH AI Center
"""
import torchvision
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, CelebA
from torchvision.transforms import functional as F
from PIL import Image
from os.path import join
import torch
import os
from ..utils.utils import set_seed
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

class ThresholdTransform(object):
  def __init__(self, thr):
    self.thr = thr # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  # do not change the data type

class TransformsSimCLR():
    """from https://github.com/alicebizeul/SimCLR/blob/master/simclr/modules/transformations/simclr.py
    A stochastic data augmentation module that transforms any given data example randomly resulting in two
    correlated views of the same example, denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, num_sim=2, jit=0.8, test=False, strength=False, ds="mnist",eval=False):
        if test: assert num_sim == 1, f"TransformsSimCLR (init): expected num_sim to be 1 for test set, got {num_sim:d}"
        self.num_sim = num_sim
        
        if eval:
            list_end=[]
            if ds == "mnist":
                list_end += [ThresholdTransform(0.5)]
            self.transform =T.Compose([T.ToTensor()]+list_end)
        else:
            list_end=[]
            list_start=[]
            scale_low=0.4
            
            if ds == "mnist":
                list_end += [ThresholdTransform(0.5)]
            if ds == "cifar10":
                list_start += [T.Resize(size=(256,256)),T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)]
                scale_low=0.6
            if ds == "celeba":
                list_start += [T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)]
                scale_low=0.6
            self.transform = T.Compose(list_start+[T.RandomResizedCrop(size=size, scale=[scale_low, 1.0], ratio=[0.75, 1.3]),T.RandomVerticalFlip(p=0.5),T.ToTensor()]+list_end)

    def __call__(self, x):
        if self.num_sim == 1:
            return self.transform(x)
        return torch.stack([self.transform(x) for _ in range(self.num_sim)], 0)

class MyCifar(torchvision.datasets.CIFAR10):

    def __init__(
        self,
        root,
        train= True,
        transform= None,
        target_transform= None,
        download= False,
    ):
        super(MyCifar,self).__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.representations={}
        self.labels={}
        self.counter=0
        self.switch=1

    def switch_off(self):
        self.switch=0
        self.data = list(self.representations.values())
        del self.representations
        self.targets = list(self.labels.values())
        del self.labels

    def set_object(self,z,labels):
        if self.switch==1:
            index=len(list(self.representations.keys()))
            for idx, (x,y) in enumerate(zip(z,labels)):
                self.representations[index+idx]=x.detach().cpu()
                self.labels[index+idx]=y.detach().cpu()
                self.counter+=1

    def __getitem__(self, index: int):

            img, target = self.data[index], self.targets[index]

            if self.switch==1:
                img = Image.fromarray(img)

                if self.transform is not None:
                    img = self.transform(img)

                if self.target_transform is not None:
                    target = self.target_transform(target)

            return img, target

class MyMNIST(torchvision.datasets.MNIST):
    def __init__(
        self,
        root,
        train=True,
        transform= None,
        target_transform=None,
        download=False,
        ):
        super(MyMNIST, self).__init__(root, train=train, download=download, target_transform=target_transform,transform=transform)
        self.switch=1
        self.representations={}
        self.labels={}
        self.counter=0

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.switch==1:
            img = Image.fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

    def switch_off(self):
        self.switch=0
        self.data = list(self.representations.values())
        del self.representations
        self.targets = list(self.labels.values())
        del self.labels
        
    def set_object(self,z,labels):
        if self.switch==1:
            index=len(list(self.representations.keys()))
            for idx, (x,y) in enumerate(zip(z,labels)):
                self.representations[index+idx]=x.detach().cpu()
                self.labels[index+idx]=y.detach().cpu()
                self.counter+=1

class MyFashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(
        self,
        root,
        train=True,
        transform= None,
        target_transform=None,
        download=False,
        ):
        super(MyFashionMNIST, self).__init__(root, train=train, download=download, target_transform=target_transform,transform=transform)
        self.switch=1
        self.representations={}
        self.labels={}
        self.counter=0

    def switch_off(self):
        self.switch=0
        self.data = list(self.representations.values())
        del self.representations
        self.targets = list(self.labels.values())
        del self.labels
        

    def set_object(self,z,labels):
        if self.switch==1:
            index=len(list(self.representations.keys()))
            for idx, (x,y) in enumerate(zip(z,labels)):
                self.representations[index+idx]=x.detach().cpu()
                self.labels[index+idx]=y.detach().cpu()
                self.counter+=1

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        if self.switch==1:
            img = Image.fromarray(img.numpy(), mode="L")

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target

class MyCelebA_eval(torchvision.datasets.CelebA):
    def __init__(self, root, transform, download=True, split=True, target_transform="hair"):
        super(MyCelebA_eval, self).__init__(root, transform=transform, download=download, split=split, target_transform=target_transform)
        self.transform_saving = T.Compose( [T.CenterCrop(64)]+ [T.ToTensor()])
        self.representations={}
        self.data_samples={}
        self.all_labels={}
        self.switch=1
        self.counter=0
        self.target_transform=target_transform
        self.fit=True

    def __getitem__(self, index: int):
        if self.switch==1:
            X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

            if self.transform is not None:
                X = self.transform(X)

            target=self.attr[index, :]
            if self.target_transform is not None:
                single_target = self.t_transform(target)

        else:
            X=self.representations[index]
            target=self.all_labels[index]
            if self.target_transform is not None:
                single_target = self.t_transform(target)

        return X, single_target #, target

    def switch_off(self):
        self.switch=0

        if self.fit :
            from sklearn.mixture import GaussianMixture
            self.representations = np.asarray([self.representations[x].detach().cpu().numpy() for x in list(self.representations.keys())])
            gmm = GaussianMixture(n_components=1, covariance_type='diag')
            gmm.fit(self.representations)
            variance = gmm.covariances_

    def switch_on(self,new_target):
        self.target_transform=new_target

    def __len__(self) -> int:
        if self.switch==1:
            return len(self.attr)
        else:
            return len(self.all_labels.keys())

    def set_object(self,z,all_labels):
        if self.switch==1:
            index=len(list(self.representations.keys()))
            for idx, (x,y) in enumerate(zip(z,all_labels)):
                # self.data_samples[index]=x.detach().cpu()
                self.representations[index+idx]=x.detach().cpu()
                self.all_labels[index+idx]=y.detach().cpu()
                self.counter+=1

    def multi_class(self,target):
        target=[target[8].item(), target[9].item(), target[11].item(), target[17].item()]
        if sum(target)==0.0: 
            target.append(1)
        else: 
            target.append(0)
        return target

    def single_class(self,target,index):
        return 1.0*(target[index].item()==1)

    def t_transform(self,target):
        if self.target_transform=="hair":
            target=self.multi_class(target)
        elif self.target_transform=="skin":
            target = self.single_class(target,26)
        elif self.target_transform=="shadow":
            target = self.single_class(target,0)
        elif self.target_transform=="a_eyebrow":
            target = self.single_class(target,1)
        elif self.target_transform=="bags":
            target = self.single_class(target,3)
        elif self.target_transform=="bald":
            target = self.single_class(target,4)
        elif self.target_transform=="bangs":
            target = self.single_class(target,5)
        elif self.target_transform=="lips":
            target = self.single_class(target,6)
        elif self.target_transform=="b_nose":
            target = self.single_class(target,7)
        elif self.target_transform=="blurry":
            target = self.single_class(target,10)
        elif self.target_transform=="b_eyebrow":
            target = self.single_class(target,12)
        elif self.target_transform=="glasses":
            target = self.single_class(target,15)
        elif self.target_transform=="makeup":
            target = self.single_class(target,18)
        elif self.target_transform=="h_cheek":
            target = self.single_class(target,19)
        elif self.target_transform=="mouth_o":
            target = self.single_class(target,21)
        elif self.target_transform=="mustache":
            target = self.single_class(target,22)
        elif self.target_transform=="n_eyes":
            target = self.single_class(target,23)
        elif self.target_transform=="gender":
            target = self.single_class(target,20)
        elif self.target_transform=="beard":
            target = self.single_class(target,24)
        elif self.target_transform=="oval":
            target = self.single_class(target,25)
        elif self.target_transform=="p_nose":
            target = self.single_class(target,27)
        elif self.target_transform=="hairline":
            target = self.single_class(target,28)
        elif self.target_transform=="r_cheek":
            target = self.single_class(target,29)
        elif self.target_transform=="sideburns":
            target = self.single_class(target,30)
        elif self.target_transform=="smile":
            target = self.single_class(target,31)
        elif self.target_transform=="s_hair":
            target = self.single_class(target,32)
        elif self.target_transform=="w_hair":
            target = self.single_class(target,33)
        elif self.target_transform=="earrings":
            target = self.single_class(target,34)
        elif self.target_transform=="hat":
            target = self.single_class(target,35)
        elif self.target_transform=="lipstick":
            target = self.single_class(target,36)
        elif self.target_transform=="young":
            target = self.single_class(target,39)
        elif self.target_transform=="all":
            target = self.multi_class(target)+[self.single_class(target,26)] + [self.single_class(target,20)]
        else: raise NotImplementedError
        return torch.tensor(target).to(DEVICE)


class MyCelebA(torchvision.datasets.CelebA):
    def __init__(self, root, transform, download=True, split=True, target_transform="hair"):
        super(MyCelebA, self).__init__(root, transform=transform, download=download, split=split, target_transform=target_transform)
        self.transform_saving = T.Compose( [T.CenterCrop(64)]+ [T.ToTensor()])
        self.representations={}
        self.labels={}
        self.switch=1
        self.fit=True

    def __getitem__(self, index: int):
        if self.switch==1:
            X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

            target: Any = []
            for t in self.target_type:
                if t == "attr":
                    target.append(self.attr[index, :])
                elif t == "identity":
                    target.append(self.identity[index, 0])
                elif t == "bbox":
                    target.append(self.bbox[index, :])
                elif t == "landmarks":
                    target.append(self.landmarks_align[index, :])
                else:
                    raise ValueError(f'Target type "{t}" is not recognized.')
            if self.transform is not None:
                X = self.transform(X)

            if target:
                target = tuple(target) if len(target) > 1 else target[0]
                if self.target_transform is not None:
                    target = self.t_transform(target)
            else:
                target = None
        else: 
            X=self.representations[index]
            target=self.labels[index]
        return X, target

    def __len__(self) -> int:
        if self.switch==1:
            return len(self.attr)
        else:
            return len(self.labels)

    def set_object(self,z,labels):
        if self.switch==1:
            index=len(list(self.representations.keys()))
            for idx, (x,y) in enumerate(zip(z,labels)):
                self.representations[index]=x.detach().cpu()
                self.labels[index]=y.detach().cpu()

    def multi_class(self,target):
        target=[target[4].item(), target[8].item(), target[9].item(), target[11].item(), target[17].item()]
        if sum(target)==0.0: target = [x/len(target) for x in target]
        else: target = [x/sum(target) for x in target]
        return target

    def single_class(self,target,index):
        return 1.0*(target[index].item()==1)

    def t_transform(self,target):
        if self.target_transform=="hair":
            target = self.single_class(target,0)
        elif self.target_transform=="hair_brown":
            target = self.single_class(target,9)
        elif self.target_transform=="hair_blond":
            target = self.single_class(target,11)
        elif self.target_transform=="hair_gray":
            target = self.single_class(target,17)
        elif self.target_transform=="skin":
            target = self.single_class(target,26)
        elif self.target_transform=="shadow":
            target = self.single_class(target,0)
        elif self.target_transform=="a_eyebrow":
            target = self.single_class(target,1)
        elif self.target_transform=="bags":
            target = self.single_class(target,3)
        elif self.target_transform=="bangs":
            target = self.single_class(target,5)
        elif self.target_transform=="lips":
            target = self.single_class(target,6)
        elif self.target_transform=="b_nose":
            target = self.single_class(target,7)
        elif self.target_transform=="blurry":
            target = self.single_class(target,10)
        elif self.target_transform=="b_eyebrow":
            target = self.single_class(target,12)
        elif self.target_transform=="glasses":
            target = self.single_class(target,15)
        elif self.target_transform=="makeup":
            target = self.single_class(target,18)
        elif self.target_transform=="h_cheek":
            target = self.single_class(target,19)
        elif self.target_transform=="mouth_o":
            target = self.single_class(target,21)
        elif self.target_transform=="mustache":
            target = self.single_class(target,22)
        elif self.target_transform=="n_eyes":
            target = self.single_class(target,23)
        elif self.target_transform=="gender":
            target = self.single_class(target,20)
        elif self.target_transform=="beard":
            target = self.single_class(target,24)
        elif self.target_transform=="oval":
            target = self.single_class(target,25)
        elif self.target_transform=="p_nose":
            target = self.single_class(target,27)
        elif self.target_transform=="hairline":
            target = self.single_class(target,28)
        elif self.target_transform=="r_cheek":
            target = self.single_class(target,29)
        elif self.target_transform=="sideburns":
            target = self.single_class(target,30)
        elif self.target_transform=="smile":
            target = self.single_class(target,31)
        elif self.target_transform=="s_hair":
            target = self.single_class(target,32)
        elif self.target_transform=="w_hair":
            target = self.single_class(target,33)
        elif self.target_transform=="earrings":
            target = self.single_class(target,34)
        elif self.target_transform=="hat":
            target = self.single_class(target,35)
        elif self.target_transform=="lipstick":
            target = self.single_class(target,36)
        elif self.target_transform=="young":
            target = self.single_class(target,39)
        elif self.target_transform=="all":
            target = self.multi_class(target)+[self.single_class(target,26)] + [self.single_class(target,20)]
        else: raise NotImplementedError
        return torch.tensor(target).to(DEVICE)

def get_loader(
    batch_size,
    data_path,
    dataset_name,
    size,
    mode="train",
    num_workers=16,
    num_sim=1,
    strength=None,
    seed=1234,
    target_transform="hair",
    eval=False,
):
    set_seed(seed)

    """Build and return a data loader:       dataset_name:  must match subdirectory name for dataset"""
    ds_path = "%s/" % (data_path)
    print(f'Looking for {dataset_name} in {ds_path}')

    if dataset_name == "cifar10":
        transform = TransformsSimCLR(size=size, num_sim=num_sim, strength=strength,ds=dataset_name,eval=eval)
        dataset = MyCifar(root=ds_path, train=(mode=="train"), download=True, transform=transform)

    elif dataset_name == "mnist":
        transform = TransformsSimCLR(size=size, num_sim=num_sim, strength=strength,ds=dataset_name,eval=eval)
        dataset = MyMNIST(root=ds_path, train=(mode=="train"), download=True, transform=transform)

    elif dataset_name == "fashionmnist":
        transform = TransformsSimCLR(size=size, num_sim=num_sim, strength=strength,ds=dataset_name,eval=eval)
        dataset = MyFashionMNIST(root=ds_path, train=(mode=="train"), download=True, transform=transform)

    elif dataset_name == "celeba":
        if eval: 
            transform = TransformsSimCLR(size=size, num_sim=num_sim, strength=strength,ds=dataset_name,eval=eval)
            dataset = MyCelebA_eval(root=ds_path, split=mode if mode=="train" else "test", download=True, transform=transform, target_transform=target_transform)
        else:
            transform = TransformsSimCLR(size=size, num_sim=num_sim, strength=strength,ds=dataset_name,eval=eval)
            dataset = MyCelebA(root=ds_path, split=mode if mode=="train" else "test", download=True, transform=transform, target_transform=target_transform)

    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        drop_last=True if (mode == "train") else False
    )
    return dataloader
