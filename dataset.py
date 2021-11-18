import json
import torch

import numpy as np

from torchvision import transforms
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, dataset, image_size, datasource, numpy_file_path, features_path=None):
        self.data = np.load(numpy_file_path)
        print('loading numpy data:', numpy_file_path)

        if features_path is not None:
            self.precomputed_features = torch.load(features_path, map_location='cpu')
            print('loading', features_path)

        self.dataset = dataset
        self.image_size = image_size
        self.ims = self.data['ims']
        self.datasource = datasource

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

        self.label_description = {
            "left": "to the left of",
            "right": "to the right of",
            "behind": "behind",
            "front": "in front of",
            "above": "above",
            "below": "below"
        }

        if dataset in ['clevr', 'igibson']:
            with open('./data/attributes.json', 'r') as f:
                data_json = json.load(f)
                self.colors_to_idx = data_json[dataset]['colors']
                self.shapes_to_idx = data_json[dataset]['shapes']
                self.materials_to_idx = data_json[dataset]['materials']
                self.sizes_to_idx = data_json[dataset]['sizes']
                self.relations_to_idx = data_json[dataset]['relations']

                self.colors = list(data_json[dataset]['colors'].keys())
                self.shapes = list(data_json[dataset]['shapes'].keys())
                self.materials = list(data_json[dataset]['materials'].keys())
                self.sizes = list(data_json[dataset]['sizes'].keys())
                self.relations = list(data_json[dataset]['relations'].keys())
        else:
            if dataset == 'visual_genome':
                selected_objects = ['man', 'person', 'woman', 'bus', 'people', 'door', 'car', 'boy', 'girl', 'lady']
                objects = {object_name: i for i, object_name in enumerate(selected_objects)}
            elif dataset == 'blocks':
                selected_objects = ['red', 'green', 'blue', 'yellow']
                objects = {object_name: i for i, object_name in enumerate(selected_objects)}
            else:
                raise NotImplementedError

            relations = {'below': 0, 'above': 1}
            self.objects = {value: key for key, value in objects.items()}
            self.relations = {value: key for key, value in relations.items()}

        self.labels = self.data['labels']
        self.size = self.labels.shape[0]

        print('image data size', self.ims.shape)
        print('label data size', self.labels.shape)
        print('resize to', (self.image_size, self.image_size))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        im = self.preprocess(self.ims[index])
        im = im.permute((1, 2, 0))
        label = self.labels[index]

        if self.datasource == 'default':
            im_corrupt = im + 0.3 * torch.randn(self.image_size, self.image_size, 3)
        elif self.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * torch.randn(self.image_size, self.image_size, 3)
        else:
            raise NotImplementedError
        return im_corrupt, im, label, self._convert_caption(label)

    def collate_fn(self, batches):
        ims_corrupt = torch.stack([x[0].float() for x in batches], dim=0)
        ims = torch.stack([x[1] for x in batches], dim=0)
        labels = [self._convert_label(x[2]) for x in batches]
        captions = [self._convert_caption(x[2]) for x in batches]
        return ims_corrupt, ims, labels, captions

    def collate_fn_clip_all(self, batches):
        ims_corrupt = torch.stack([x[0].float() for x in batches], dim=0)
        ims = torch.stack([x[1] for x in batches], dim=0)
        labels = torch.cat([self._convert_to_sentence(x[2]) for x in batches], dim=0)
        captions = [self._convert_caption(x[2]) for x in batches]
        return ims_corrupt, ims, labels, captions

    def _convert_to_sentence(self, label):
        if self.dataset == 'blocks':
            obj1 = self.objects[label[0]]
            relation = self.relations[label[1]]
            obj2 = self.objects[label[2]]
            return torch.as_tensor(self.precomputed_features[f'{obj1} {relation} {obj2}'])
        else:
            text_label = []
            for i in range(2):
                shape, size, color, material, pos = label[i * 5:i * 5 + 5]
                text_label.append(' '.join([self.sizes[size], self.colors[color],
                                            self.materials[material], self.shapes[shape]]))

            relation = self.relations[label[-1]]

            if relation == 'none':
                sentence = f'{text_label[0]}'
                return self.precomputed_features[sentence]
            else:
                sentence = f'{text_label[0]} {self.label_description[relation]} {text_label[1]}'
                return self.precomputed_features[sentence]

    def _convert_label(self, label):
        """
            convert label to return
            (first object text description, second object text description, relation, 0, 1/2)
            where 0 encodes the position of first object, 1 represents the position of second objects if exists.
        """
        if self.dataset == 'visual_genome' or self.dataset == 'blocks':
            obj1 = self.objects[label[0]]
            obj2 = self.objects[label[2]]
            return self.precomputed_features[obj1], self.precomputed_features[obj2], label[1], 0, 1
        else:
            text_label = []
            positions = []
            for i in range(2):
                shape, size, color, material, pos = label[i*5:i*5+5]
                text_label.append(' '.join([self.sizes[size], self.colors[color],
                                            self.materials[material], self.shapes[shape]]))
                positions.append(pos)

            relation = self.relations[label[-1]]

            # single object
            if relation == 'none':
                return self.precomputed_features[text_label[0]], \
                       self.precomputed_features[''], \
                       label[-1], positions[0], positions[1]
            else:
                return self.precomputed_features[text_label[0]], \
                       self.precomputed_features[text_label[1]], \
                       label[-1], positions[0], positions[1]

    def _convert_caption(self, label):
        if self.dataset == 'blocks':
            return f'{self.objects[label[0]]} {self.relations[label[1]]} {self.objects[label[2]]}'
        else:
            text_label = []
            for i in range(2):
                shape, size, color, material, pos = label[i * 5:i * 5 + 5]
                obj = ' '.join([self.sizes[size], self.colors[color],
                                self.materials[material], self.shapes[shape]]).strip()
                text_label.append(obj)
            relation = self.relations[label[-1]]
            if relation == 'none':
                return text_label[0]
            else:
                return f'{text_label[0]} {self.label_description[relation]} {text_label[1]}'
