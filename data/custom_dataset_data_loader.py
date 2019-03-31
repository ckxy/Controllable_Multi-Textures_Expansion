import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'star' or opt.dataset_mode == 'n2':
        from data.half_dataset import N2Dataset
        dataset = N2Dataset()
    elif opt.dataset_mode == 'star1' or opt.dataset_mode == 'n2t':
        from data.half_dataset import N2TestDataset
        dataset = N2TestDataset()
    elif opt.dataset_mode == 'star2' or opt.dataset_mode == '2n':
        from data.half_dataset import _2NDataset
        dataset = _2NDataset()
    elif opt.dataset_mode == 'single_star' or opt.dataset_mode == 'sn2t':
        from data.single_dataset import SingleN2Dataset
        dataset = SingleN2Dataset()
    elif opt.dataset_mode == 'single_star1' or opt.dataset_mode == 'snt':
        from data.single_dataset import SingleNDataset
        dataset = SingleNDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
