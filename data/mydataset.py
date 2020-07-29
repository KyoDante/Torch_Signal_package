from torch.utils.data import Dataset


class trainDataset(Dataset):
    def __init__(self):
        # constructor
        self.src_dir = './'
        self.train_set = []
        self.test_set = []
        self.valid_set = []
        self.domains = []
        self.classes = []
        pass

    def __getitem__(self, index):
        # 对应Index号的数据和标签
        pass

    def __len__(self):
        pass

    