import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset


def make_split_by_domain(data: np.ndarray, label: np.ndarray, 
                         domain: np.ndarray, train_domain: list, test_domain: list,
                         batch_size = 64):
    
    # 对域的字符串重新编码
    domain_code = {}
    domain_num = 0
    for d in np.unique(domain):
        domain_code[d] = domain_num
        domain_num += 1
    
    # domain[domain==]
    
    train_domain_data = []
    train_domain_label = []
    test_domain_data = []
    test_domain_label = []
    
    
    train_set = TensorDataset(train_domain_data, train_domain_label)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_set = TensorDataset(test_domain_data, test_domain_label)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, test_loader


if __name__ == "__main__":
    data = np.load("data.npy")
    label = np.load("label.npy")
    domain = np.load("domain.npy")
    
    import torchaudio as ta
    ta.transforms.MFCC(44100)