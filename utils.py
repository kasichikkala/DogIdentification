from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_paths=None, labels=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        classes_dir_names = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_label = {class_name: label for label, class_name in enumerate(classes_dir_names)}
        self.classes = [d.split('-')[1] for d in classes_dir_names]

        if img_paths is None or labels is None:

            for class_name in classes_dir_names:
                class_path = os.path.join(root_dir, class_name)
                label = self.class_to_label[class_name]

                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.data.append((img_path, label))
        else:

            for img_path, label in zip(img_paths, labels):
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]


        img = Image.open(img_path).convert('RGB')

        if self.transform:
            # print(np.array(img).shape)
            img = self.transform(img)

        return img, label


def visualize_batch(train_loader, train_dataset, num_images=9):

    original_mean = [0.485, 0.456, 0.406]
    original_std = [0.229, 0.224, 0.225]


    unnormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip(original_mean, original_std)],
        std=[1 / s for s in original_std]
    )


    train_img, train_label = next(iter(train_loader))
    # shape, img=[batch_size, 3, 224, 224], label=[batch_size]
    print(train_img.shape, train_label.shape)


    img = train_img[:num_images]  # [num_images, 3, 224, 224]

    img = unnormalize(img)

    img = img.numpy()
    class_label = train_label.numpy()

    img = np.transpose(img, [0, 2, 3, 1])

    plt.figure()

    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img[i])
        plt.xticks([])
        plt.yticks([])
        plt.title(train_dataset.classes[class_label[i]])

    plt.tight_layout()
    plt.show()
