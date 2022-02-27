import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import torch.utils.data as util_data
from torch.utils.data import Dataset 
from Estimator import Estimator

!git clone https://github.com/dsfsi/textaugment.git    
!pip install textaugment 
  
from textaugment import EDA                     
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

t1 = EDA(random_state=1)  
t2 = EDA(random_state=2)
def transf1(s): 
    return t1.random_deletion(t1.random_insertion(t1.synonym_replacement(s)),p=0.1)
def transf2(s):
    return t2.random_deletion(t2.random_insertion(t2.synonym_replacement(s)),p=0.1) 

#data loading
class args1:
    text='text' 
    label='topic' 
    batch_size=8     
    augmentation_1='text_aug_one'
    augmentation_2='text_aug_two'
train=pd.read_csv('twitter_aug_train.csv')
valid=pd.read_csv('twitter_aug_val.csv')
test=pd.read_csv('twitter_aug_test.csv')

#some preprocessing  
le=LabelEncoder() 
train[args.label]=le.fit_transform(train[args.label])
valid[args.label]=le.fit_transform(valid[args.label])
test[args.label]=le.fit_transform(test[args.label]) 
  
#augmentation
train_aug=train.copy()
valid_aug=valid.copy()
test_aug=test.copy()
def augment(df):
    for i in range(df.shape[0]):
         df[args.augmentation_1][i]=transf1(df[args.text][i])
         df[args.augmentation_2][i]=transf2(df[args.text][i])
         df[args.text][i]=df[args.text][i].lower()
         df[args.augmentation_1][i]=df[args.augmentation_1][i].lower()
         df[args.augmentation_2][i]=df[args.augmentation_2][i].lower()
    return df
train_aug=augment(train_aug)
valid_aug=augment(valid_aug)
test_aug=augment(test_aug)

#DataLoader  
class VirtualAugSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

    
class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx], 'label': self.train_y[idx]}
       

def explict_augmentation_loader(args):
    train_data = train_aug
    train_text = train_data[args.text].fillna('.').values
    train_text1 = train_data[args.augmentation_1].fillna('.').values
    train_text2 = train_data[args.augmentation_2].fillna('.').values
    train_label = train_data[args.label].astype(int).values  

    train_dataset = ExplitAugSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


def unshuffle_val_loader(args):
    train_data = valid_aug

    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader   
def unshuffle_test_loader(args):
    train_data = test_aug
  
    train_text = train_data[args.text].fillna('.').values  
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)     
    return train_loader   
