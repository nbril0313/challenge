from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from augmentation import CIFAR10Policy  

class CustomDataset(Dataset):
    def __init__(self, images, labels=None, mode='train'):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.transform = self.load_transforms()
        
    def load_transforms(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                CIFAR10Policy(),  
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image
