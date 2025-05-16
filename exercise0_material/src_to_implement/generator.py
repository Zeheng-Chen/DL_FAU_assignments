import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        # read label
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)

        # get image file
        self.image_files = [f for f in os.listdir(self.file_path) if f.endswith('.npy')]

        # ini index
        self.indices = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.current_index = 0
        self.epoch = 0
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        image, label = [], []
        while(len(image) < self.batch_size):
            if self.current_index >= len(self.indices):
                self.epoch += 1
                self.current_index = 0
                if self.shuffle:
                    np.random.shuffle(self.indices)
            idx = self.indices[self.current_index]
            self.current_index += 1

            img = np.load(os.path.join(self.file_path, self.image_files[idx]))
            img = self.augment(img)
            img = resize(img, self.image_size, preserve_range=True).astype(np.uint8)
            label_key = self.image_files[idx].replace('.npy', '')
            image.append(img)
            label.append(self.labels[label_key])

            # 最终一定凑到 batch_size
        return np.stack(image), np.array(label, dtype=np.int64)

        # if self.current_index + self.batch_size > len(self.image_files):
        #     # when over the size of image next epoch
        #
        #     self.epoch += 1
        #     self.current_index = 0
        #     if self.shuffle:
        #         np.random.shuffle(self.indices)
        #
        # batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        # self.current_index += self.batch_size
        #
        # images = []
        # labels = []
        # for idx in batch_indices:
        #     img = np.load(os.path.join(self.file_path, self.image_files[idx]))
        #     img = self.augment(img)
        #     # img = resize(img, self.image_size, preserve_range=True)
        #     img = resize(img, self.image_size, preserve_range=True).astype(np.uint8)
        #     label = self.labels[self.image_files[idx].replace('.npy', '')]
        #     images.append(img)
        #     labels.append(label)

        # return np.stack(image), np.array(label, dtype=np.int64)



    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if self.mirroring and np.random.rand() > 0.5:
            flip_mode = np.random.choice(['h', 'v', 'hv'])
            if flip_mode == 'h':
                img = np.fliplr(img)
            elif flip_mode == 'v':
                img = np.flipud(img)
            else:  # 'hv'
                img = np.flipud(np.fliplr(img))
        if self.rotation:
            k = np.random.choice([1, 2, 3])  # 保证真正旋转
            img = np.rot90(img, k)  # rot90 返回新数组，形状自动调整

        return img


    def current_epoch(self):
        # return the current epoch number
        return self.epoch


    def class_name(self, label):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return class_names[label]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()
        # 假设我们要在一行显示5个图像，可以根据图像数量计算需要多少行
        num_images = len(images)
        num_columns = 5
        num_rows = (num_images + num_columns - 1) // num_columns  # 计算所需行数

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2.5))  # 调整画布大小
        fig.subplots_adjust(hspace=0.4, wspace=0.4)  # 调整子图之间的间距

        for i, (img, label) in enumerate(zip(images, labels)):
            ax = axes[i // num_columns, i % num_columns]  # 选择正确的子图
            ax.imshow(img)
            ax.set_title(self.class_name(label), fontsize=10)  # 设置较小的字体大小
            ax.axis('off')

        # 隐藏多余的子图
        for j in range(i + 1, num_rows * num_columns):
            axes[j // num_columns, j % num_columns].axis('off')

        plt.show()






