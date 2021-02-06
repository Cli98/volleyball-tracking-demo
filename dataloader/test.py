import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


class freespace(data.Dataset):
    def __init__(self, opt):
        super(freespace, self).__init__()
        self.image_list = []
        self.root = ""
        self.affix = "png"
        self.mode = "train"
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_labels = 2
        self.__setitem__(self.opt)

    def __setitem__(self, dict):
        self.root = dict.dataroot
        self.affix = dict.affix
        self.mode = dict.phase
        self.compact_size = (dict.useWidth, dict.useHeight)
        assert self.compact_size[0] % 32 == 0 and self.compact_size[
            1] % 32 == 0, "input size should be dividable by factor of 32!"
        if dict.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', 'image_2', '*.png')))
        elif dict.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', 'image_2', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', 'image_2', '*.png')))

    def __getitem__(self, index):
        color_image = cv2.cvtColor(cv2.imread(self.image_list[index]), cv2.COLOR_BGR2RGB)
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]
        height, width, depth = color_image.shape
        label = np.zeros((height, width), dtype=np.uint8)
        if self.mode.lower() in ["train", "val"]:
            if not os.path.exists(
                    os.path.join(useDir, 'gt_image_2', name[:-10]+'road_'+name[-10:])):
                print("label does not exist! please check whether the path is correct!")
                exit(-1)
            gt_image = cv2.cvtColor(
                cv2.imread(os.path.join(useDir, 'gt_image_2', name[:-10]+'road_'+name[-10:])),
                cv2.COLOR_BGR2RGB)
            label[gt_image[:, :, 2] > 0] = 1

        compact_color_image = cv2.resize(color_image, self.compact_size, interpolation=cv2.INTER_CUBIC)
        compact_color_image = transforms.ToTensor()(compact_color_image)

        compact_label = cv2.resize(label, self.compact_size, interpolation=cv2.INTER_CUBIC)
        compact_label[compact_label > 0] = 1
        compact_label = torch.from_numpy(compact_label)
        compact_label = compact_label.type(torch.LongTensor)

        return {'rgb_image': compact_color_image, 'label': compact_label,
                'path': name, 'oriSize': (width, height)}

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    # Compact_size: width, height
    opt = {"path": os.path.join(".", "dataloader", "kitti", 'training', 'image_2'), "affix": "png",
           "mode": "train", "size": (1248, 384)}
    dataset = freespace(opt)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for item in dataloader:
        color_image, label, path, oriSize = np.transpose(np.squeeze(item["color_image"].numpy(), axis=0),
                                                         (1, 2, 0)), np.squeeze(item["label"].numpy(), axis=0), item[
                                                "path"], item["oriSize"]
        plt.imshow(color_image)
        plt.imshow(label, alpha=0.3)
        plt.axis('off')
        plt.show()
        plt.pause(0.5)
        print("The path is: {} \n".format(path[0]))
