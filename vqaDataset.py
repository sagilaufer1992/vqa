from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
from skimage import io

image_prefix = "COCO_train2014_"


class VqaDataset(Dataset):
    def __init__(self, images_path, questions_path, annotations_path):
        self.images_path = images_path
        self.questions = self.__loadJson(questions_path)
        self.annotations = self.__loadJson(annotations_path)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224])])  

    def __len__(self):
        return len(self.questions['questions'])

    def __getitem__(self, idx):
        idx = idx % 100 # TODO: remove!!!!!! when training whole data
        annot = self.annotations['annotations'][idx]
        question = self.questions['questions'][idx]
        img_id = str(annot['image_id'])
        img_path = "{path}{prefix}{imagesindex}.jpg".format(path=self.images_path, 
                                                            prefix=image_prefix, 
                                                            imagesindex=img_id.zfill(12))
        image = io.imread(img_path)         

        return self.transform(image), annot, question['question']

    def __loadJson(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
