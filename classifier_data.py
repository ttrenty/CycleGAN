import os
import glob
import time

import random

from torch.utils.data import Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



class BinaryClassificationImageDataset(Dataset):
    def __init__(self, root, transformations=None, mode="train"):
        self.transform = transforms.Compose(transformations)

        self.files_A = sorted(glob.glob(os.path.join(root, mode, "A", "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(root, mode, "B", "*.*")))

        self.length = len(self.files_A) + len(self.files_B)

    def __getitem__(self, index):
        if index < len(self.files_A):
            img_path = self.files_A[index]
            lbl = 0
        else:
            img_path = self.files_B[index - len(self.files_A)]
            lbl = 1

        img = Image.open(img_path)

        return self.transform(img), lbl

    def __len__(self):
        return self.length


def tensor_to_image(tensor_image):
    image = tensor_image.detach().to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def show_sample(dataset, class_a_name, class_b_name):
    """Show 10 examples of the dataset and their labels.
    Display the image and the label in one figure
    """
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, j in enumerate(random.sample(range(len(dataset)), 10)):
        img, lbl = dataset[j]
        ax = axs[i // 5, i % 5]
        ax.imshow(tensor_to_image(img))
        ax.set_title(class_a_name if lbl == 0 else class_b_name)
        ax.axis("off")
    plt.show()


if __name__ == '__main__':
    img_size = 64

    transformations = [
        transforms.Resize(int(img_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    train_dataset = BinaryClassificationImageDataset(
        os.path.join(".", "datasets", "apple2orange64"),
        transformations=transformations,
        mode="train"
    )

    # measure mean time for loading an image from the dataset
    total_time = 0.
    for i in range(len(train_dataset)):
        tic = time.time()
        img, lbl = train_dataset[i]
        toc = time.time()
        total_time += toc - tic
    mean_time = total_time / len(train_dataset)
    print(f"Mean time for loading an image: {mean_time:.10f} sec")

    show_sample(train_dataset, 'apple', 'orange')

    # from torch.utils.data import DataLoader

    # data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # batch = next(iter(data_loader))
    # print(f"{batch[0].size() = }")
    # print(f"{batch[1].size() = }")
