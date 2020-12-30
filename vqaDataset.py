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
        self.possible_answers = self.__init_possible_answers()

        self.relevant_indices = [True] * len(self.questions['questions'])
        self.annotaions_scores = self.__init_annotaions_scores_and_relevant_indices()

        self.temp_questions = []
        self.temp_annotation_scores = []
        self.temp_annotations = []

        for index, relevance in enumerate(self.relevant_indices):
            if relevance:
                self.temp_questions.append(self.questions['questions'][index])
                self.temp_annotation_scores.append(self.annotaions_scores[index])
                self.temp_annotations.append(self.annotations['annotations'][index])

        self.questions = self.temp_questions
        self.annotaions_scores = self.temp_annotation_scores
        self.annotations = self.temp_annotations


    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        annot = self.annotations[idx]
        question = self.__translate_question(self.questions[idx]['question'])
        img_id = str(annot['image_id'])
        img_path = "{path}{prefix}{imagesindex}.jpg".format(path=self.images_path, 
                                                            prefix=image_prefix, 
                                                            imagesindex=img_id.zfill(12))
        annotation_score = self.annotaions_scores[idx]
        image = io.imread(img_path)
        transformed_image = self.transform(image)

        # in case of B&W pictures
        print (transformed_image.size()[0], img_id)
        if transformed_image.size()[0] == 1:
            print("dfafasdf", idx)
            transformed_image = torch.stack([transformed_image, transformed_image.clone().detach(), transformed_image.clone().detach()])
            print ("qweqweqweqwe", transformed_image.size()[0])


        return self.transform(image), annotation_score, question

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

    def __init_possible_answers(self):
        word_hist = dict({})
        for a in self.annotations['annotations']:
            ans = a['multiple_choice_answer']
            if ans in word_hist:
                word_hist[ans] = word_hist[ans] + 1
            else:
                word_hist[ans] = 1

        list_top_words = list(reversed(sorted(word_hist.items(), key=lambda item: item[1])))[0:128]
        top_words = {word: i for i, word in enumerate(list(map(lambda x: x[0], list_top_words)))}
        return top_words

    def __init_annotaions_scores_and_relevant_indices(self):
        scores = []
        for i, a in enumerate(self.annotations['annotations']):
            answers = a['answers']
            score_dict = dict({})
            for ans in answers:
                if ans['answer'] in score_dict:
                    score_dict[ans['answer']] += 1
                else:
                    score_dict[ans['answer']] = 1
            score_vec = torch.zeros(128)

            is_relevant = False

            for key in score_dict:
                if key in self.possible_answers:
                    is_relevant = True
                    score_vec[self.possible_answers[key]] = min(score_dict[key] / 3, 1)
            
            if not is_relevant:
                self.relevant_indices[i] = False

            # print(score_vec)
            scores.append(score_vec)

        return scores