import random
import os

from PIL import Image
from torch.utils.data import Dataset


class DesiredImageNetData(Dataset):
    """
    root_dir: Dataset directory location
    num_images_per_class: Number of images per class (Randomly)
    transform: Apply Transformation (Default: None)
    """
    def __init__(self, root_dir='<replace-path-to-imagenet-folders>',
                 num_images_per_class=500, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        random.shuffle(self.classes)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images, self.class_names = self._load_images(num_images_per_class)

    def _load_images(self, num_images_per_class):
        images = []
        class_names = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            class_names.append(class_dir.split('_')[-1])
            filenames = os.listdir(class_dir)
            random.shuffle(filenames)
            filenames = filenames[:num_images_per_class]
            for filename in filenames:
                if filename.endswith('.jpg'):
                    image_path = os.path.join(class_dir, filename)
                    images.append((image_path, self.class_to_idx[cls]))
        return images, class_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
