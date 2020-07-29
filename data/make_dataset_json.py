import pandas as pd
from tqdm import tqdm as tqdm
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
import torch


# src_dir: "D:/AcouDigits/journalExtension/AcouDigits_图片数据/预处理后原始图片数据/第一次数字实验"
# domains = ['DQL','HSC', 'HYT', 'WD', 'YQ','ZM']
# classes = ['0','1','2','3','4','5','6','7','8','9']
def make_dataset_json(src_dir='D:/AcouDigits/journalExtension/AcouDigits_图片数据/预处理后原始图片数据/第一次数字实验',
                      domains=['DQL','HSC', 'HYT', 'WD', 'YQ','ZM'],
                      classes=['0','1','2','3','4','5','6','7','8','9']):
    transform = Compose(
        [
            Resize((224,224)),
            ToTensor(),
        ]
    )
    
    images = []
    labels = []
    domain_labels = []

    for domain in domains: # 人的域
        domain_dir = os.path.join(src_dir, domain)
        for char in classes: # 字符的类
            char_dir = os.path.join(domain_dir, char)
            for img in os.listdir(char_dir): # 单个图片sample
                img_dir = os.path.join(char_dir, img)
                image = Image.open(img_dir, mode='r')
                image_data = transform(image)
                # labels.append(int(char))
                # images.append(image_data.numpy())
                domain_labels.append(domain)

    # np.save("data.npy",np.array(images))
    # np.save("label",np.array(labels))
    np.save("domain",np.array(domain_labels))

    # json_data = pd.DataFrame()
    # json_data['data'] = images
    # json_data['label'] = labels
    # json_data['domain'] = domain_labels
    # json_data.to_json('acouDigits_224.json')

    # ImportError: Missing optional dependency 'tables'.  Use pip or conda to install tables.
    # OverflowError: Python int too large to convert to C long
    # json_data.to_hdf('datasets.h5',key='AcouDigits')


if __name__ == "__main__":
    make_dataset_json()

    domains = np.load("data.npy")
    print(domains.shape)

    # data = pd.read_hdf('datasets.h5',key='AcouDigits')
    # print(data.head())