import os
import numpy as np

from jittor import dataset


class FlatDirectoryImageDataset(dataset.Dataset):
    """ jittor Dataset wrapper for the generic flat directory images dataset """

    def __setup_files(self):
        """
        private helper for setting up the files_list
        :return: files => list of paths of files
        """
        file_names = os.listdir(self.data_dir)
        files = []  # initialize to empty list

        for file_name in file_names:
            possible_file = os.path.join(self.data_dir, file_name)
            if os.path.isfile(possible_file):
                files.append(possible_file)

        # return the files list
        return files

    def __init__(self, data_dir, transform=None):
        """
        constructor for the class
        :param data_dir: path to the directory containing the data
        :param transform: transforms to be applied to the images
        """
        super().__init__()
        # define the state of the object
        self.data_dir = data_dir
        self.transform = transform

        # setup the files for reading
        self.files = self.__setup_files()
        self.total_len = len(self.files)
        # this function must be called
        self.set_attrs(total_len = self.total_len)
    '''
    def __len__(self):
        """
        compute the length of the dataset
        :return: len => length of dataset
        """
        return len(self.files)
    '''
    def __getitem__(self, idx):
        """
        obtain the image (read and transform)
        :param idx: index of the file required
        :return: img => image array
        """
        from PIL import Image

        img_file = self.files[idx]

        if img_file[-4:] == ".npy":
            # files are in .npy format
            img = np.load(img_file)
            img = Image.fromarray(img.squeeze(0).transpose(1, 2, 0))

        else:
            # read the image:
            img = Image.open(self.files[idx]).convert('RGB')

        # apply the transforms on the image
        if self.transform is not None:
            img = self.transform(img)

        if img.shape[0] >= 4:
            # ignore the alpha channel
            # in the image if it exists
            img = img[:3, :, :]

        # return the image:
        return img, 0

