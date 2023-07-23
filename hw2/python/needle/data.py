import numpy as np
from .autograd import Tensor
import gzip
import struct

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            img = np.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        new_img = np.zeros((H + 2 * self.padding, W + 2 * self.padding, C))
        new_img[self.padding:self.padding + H, self.padding: self.padding + W, :] = img
        new_img = new_img[self.padding + shift_x: self.padding + shift_x + H, self.padding + shift_y: self.padding + shift_y + W, :]
        return new_img
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        else:
            shuff_arr = np.arange(len(dataset))
            np.random.shuffle(shuff_arr)
            self.ordering = np.array_split(shuff_arr,
                                           range(batch_size, len(dataset), batch_size))
        self.index = 0

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        return self
        ### END YOUR SOLUTION

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index == len(self.ordering):
            raise StopIteration
        index = self.index
        self.index += 1
        data_index = self.ordering[index]
        batch = self.dataset[data_index]
        return [Tensor(x) for x in batch]
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        img = np.reshape(img, (28, 28, -1))
        img = self.apply_transforms(img)
        img = np.reshape(img, (-1, 28 * 28))
        lab = self.labels[index]
        return img, lab
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

def parse_mnist(image_filename, label_filename):
    with gzip.open(image_filename, 'rb') as img_file:
        magic, numbers, row, col = struct.unpack(">4I", img_file.read(16))
        images = np.fromstring(img_file.read(), dtype=np.uint8).reshape(numbers, 784).astype('float32') / 255

    with gzip.open(label_filename, 'rb') as lab_file:
        magic, numbers = struct.unpack(">2I", lab_file.read(8))
        labels = np.fromstring(lab_file.read(), dtype=np.uint8)
    return (images, labels)