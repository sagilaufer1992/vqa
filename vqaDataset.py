import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
from skimage import io
from torch.nn.utils.rnn import pad_sequence

image_prefix = "COCO_train2014_"

class VqaDataset(Dataset):
    def __init__(self, images_path, questions_path, annotations_path):
        self.images_path = images_path
        self.questions = self.__loadJson(questions_path)
        self.annotations = self.__loadJson(annotations_path)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([224, 224])])

        self.words_dictionary = self.__init_word_dictionary()

    def __len__(self):
        return len(self.questions['questions'])

    def __getitem__(self, idx):
        idx = idx % 100 # TODO: remove!!!!!! when training whole data
        annot = self.annotations['annotations'][idx]
        question = self.__translate_question(self.questions['questions'][idx]['question'])
        img_id = str(annot['image_id'])
        img_path = "{path}{prefix}{imagesindex}.jpg".format(path=self.images_path, 
                                                            prefix=image_prefix, 
                                                            imagesindex=img_id.zfill(12))
        image = io.imread(img_path)         

        return self.transform(image), annot, question

    def __loadJson(self, path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data

    def __init_word_dictionary(self):
        word_hist = dict({})
        for q in self.questions['questions']:
            for w in q['question'].replace("?", "").replace("'s", "").lower().split(' '):
                if len(w) == 0:
                    continue
                if w in word_hist:
                    word_hist[w] = word_hist[w] + 1
                else:
                    word_hist[w] = 1
        list_top_words = list(reversed(sorted(word_hist.items(), key=lambda item: item[1])))[0:1022]
        top_words = {word: i for i, word in enumerate(list(map(lambda x: x[0], list_top_words)))}

        return top_words

    def __translate_question(self, q):
        pad = np.full(max(1, 16 - len(q.split(' '))), 1023)
        a = torch.tensor(list(map(self.__translate_word, q.replace("?", "").replace("'s", "").lower().split(' ')[0:15])))
        a = torch.cat([a, torch.tensor(pad)])
        return a
    
    def __translate_word(self, w):
        if w in self.words_dictionary:
            return self.words_dictionary[w]
        return 1022