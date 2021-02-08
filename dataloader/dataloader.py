import torch.utils.data as data
import torch
import os
from glob import glob
import cv2
from sklearn.model_selection import train_test_split
import numpy as np


class Volleyball_loader(data.Dataset):
    def __init__(self, root, affix, scale=None, mode="train", split_ratio=0.1, seed=2020):
        self.root = root
        self.image_list = None
        self.affix = affix
        self.need_resize = False
        self.mode = mode
        self.split_ratio = split_ratio
        self.seed = seed
        if scale:
            self.need_resize = True
            self.target_height = scale[0]
            self.target_width = scale[1]
        self.collect_data()

    def __getitem__(self, index):
        img_path, cid = self.image_list[index]
        image = cv2.imread(img_path)
        if self.need_resize:
            image = cv2.resize(image, (self.target_width, self.target_height), cv2.INTER_CUBIC)
        return torch.from_numpy(np.transpose(image, [2,0,1])), torch.tensor(cid)

    def __len__(self):
        return len(self.image_list)

    def collect_data(self):
        if not os.path.exists(self.root):
            print("Data folder not exists!")
            return
        pos_folder_path = os.path.join(self.root, "ball", "*." + self.affix)
        neg_folder_path = os.path.join(self.root, "background", "*." + self.affix)
        pos_image_list = [(file, 1) for file in glob(pos_folder_path)]
        neg_image_list = [(file, 0) for file in glob(neg_folder_path)]
        image_list = pos_image_list + neg_image_list
        assert len(image_list)>0, "No image data found!"
        if self.mode == "train":
            self.image_list, _ = train_test_split(image_list, test_size=self.split_ratio, shuffle=True,
                                                  random_state=self.seed)
        else:
            _, self.image_list = train_test_split(image_list, test_size=self.split_ratio, shuffle=True,
                                                  random_state=self.seed)
        assert len(self.image_list) > 0, "No data gathered for this task!"

    def Visualize_data(self, item):
        image, cid = item
        cv2.imshow("volleyball_{}".format(cid.numpy()), image.numpy().squeeze())
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


if __name__ == "__main__":
    affix = "jpg"
    root = "./data"
    dataset = Volleyball_loader(root, affix)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for item in dataloader:
        dataset.Visualize_data(item)
