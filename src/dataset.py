import os
import re
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class VerifyDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir, mode, height=48, width=128):
        self.height = height
        self.width = width
        self.paths, self.texts = self.load_from_dir(root_dir, mode)

    def load_from_dir(self, dir, mode):
        print("loading data:", end=" ")
        dir += "/" + mode

        paths = []
        texts = []
        pattern = re.compile(r'\d+_(.{4}).jpg')
        for name in os.listdir(dir):
            matchObj = pattern.match(name)
            if matchObj:
                paths.append(os.path.join(dir, name))
                texts.append(matchObj.group(1))

        print(len(paths), "张图片")
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.crop((40, 2, 168, 50))
        image = np.array(image)
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)
        image = image.unsqueeze(0)

        target = [self.CHAR2LABEL[c] for c in self.texts[index]]
        target = torch.LongTensor(target)

        target_length = torch.LongTensor([4])

        return image, target, target_length


def verify_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths
