import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader

from args import get_args
from tps_transformation import tps_transform
from image_utils import appearence_transformation, xdog



class AnimeDataSet(Dataset):

    def __init__(self, args, img_transform_gt, img_transform_sketch):
        self.args = args
        self.img_transform_gt = img_transform_gt
        self.img_transform_sketch = img_transform_sketch
        self.img_dir = os.path.join(args.train_dir)
        self.sket_dir = os.path.join(args.train_sket_dir)

        self.data_list = glob.glob(os.path.join(self.img_dir, '*.png'))
        self.data_list = [x.split(os.sep)[-1].split('_')[0] for x in self.data_list]
        self.data_list = list(set(self.data_list))

    def __getitem__(self, index):
        fid = self.data_list[index]
        reference = Image.open(os.path.join(self.img_dir, fid)).convert('RGB')
        sketch = Image.open(os.path.join(self.sket_dir, fid)).convert('L')
        # sketch = xdog(sketch)
        
        reference = appearence_transformation(self.args, reference)
        reference = Image.fromarray(reference.astype('uint8'))

        augmented_reference = tps_transform(np.array(reference))
        
        return {
            'Ir':self.img_transform_gt(Image.fromarray(augmented_reference)), 
            'Igt':self.img_transform_gt(reference), 
            'Is':self.img_transform_sketch(sketch)
        }
        
    def __len__(self):
        return len(self.data_list)


def get_loader(args):

    img_size = args.img_size
    
    color_transform = T.Compose([
        # T.ColorJitter(brightness=0.05, contrast=0.1, saturation=0.2, hue=0.2),
        # T.RandomHorizontalFlip(p=0.5),
        T.Resize((img_size, img_size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    
    sketch_transform = T.Compose([
        # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # T.RandomHorizontalFlip(p=0.5),
        T.Resize((img_size, img_size), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize(mean=(0.5), std=(0.5))
    ])

    dataset = AnimeDataSet(args, color_transform, sketch_transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             drop_last=True)
    return data_loader



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    args = get_args()
    loader = get_loader(args)
    for batch in loader:
        Ir = batch['Ir']
        Igt = batch['Igt']
        Is = batch['Is']
        plt.imshow(Ir[0].permute(1,2,0))
        plt.show()
        plt.imshow(Igt[0].permute(1,2,0))
        plt.show()
        plt.imshow(Is[0].permute(1,2,0), cmap='gray')
        plt.show()
        