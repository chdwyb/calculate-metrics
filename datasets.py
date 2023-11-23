import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf


class MyDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain):
        super(MyDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)

        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)


    def __len__(self):
        return len(self.inputImages)

    def __getitem__(self, index):

        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])
        inputImage = Image.open(inputImagePath).convert('RGB')

        targetImagePath = os.path.join(self.targetPath, self.inputImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target