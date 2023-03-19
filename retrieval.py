import json
import random
import argparse

import torch
import clip
import numpy as np

from PIL import Image
from itertools import permutations
from models import ResNetModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load_clip(clip_checkpoint_path=None):
    model, preprocess = clip.load("ViT-B/32", device=device)
    input_resolution = model.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    if clip_checkpoint_path is not None:
        checkpoint = torch.load(clip_checkpoint_path, map_location='cpu')
        checkpoint['model_state_dict']['input_resolution'] = input_resolution
        checkpoint['model_state_dict']['context_length'] = context_length
        checkpoint['model_state_dict']['vocab_size'] = vocab_size
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loading fine-tuned CLIP from: {clip_checkpoint_path}')

    model.eval()
    return model, preprocess


def load_ebm(ebm_checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(ebm_checkpoint_path, map_location='cpu')
    FLAGS = checkpoint['FLAGS']
    model = ResNetModel(FLAGS).to(device).eval()
    return model


def rank_ebm_captions(image_path, ebm_checkpoint_path=None):
    with open('./data/attributes.json', 'r') as f:
        data_json = json.load(f)
        description = {
            "left": ["to the left of"],
            "right": ["to the right of"],
            "behind": ["behind"],
            "front": ["in front of"],
            "above": ["above"],
            "below": ["below"]
        }

        dataset = args.dataset
        colors_to_idx = data_json[dataset]['colors']
        shapes_to_idx = data_json[dataset]['shapes']
        materials_to_idx = data_json[dataset]['materials']
        sizes_to_idx = data_json[dataset]['sizes']
        relations_to_idx = data_json[dataset]['relations']

        idx_to_colors = list(colors_to_idx.keys())
        idx_to_shapes = list(shapes_to_idx.keys())
        idx_to_materials = list(materials_to_idx.keys())
        idx_to_sizes = list(sizes_to_idx.keys())
        idx_to_relations = list(relations_to_idx.keys())

    def object_label(obj):
        size, color, material, shape = obj
        if args.generalization:
            return [shapes_to_idx['none'], sizes_to_idx['none'], colors_to_idx[color], materials_to_idx['none']]
        return [shapes_to_idx[shape], sizes_to_idx[size], colors_to_idx[color], materials_to_idx[material]]

    def construct_label(objs, rels):  # 4x11 - where 11 is the size of a single label
        numeric_label = []
        for i in range(len(objs) - 1):
            numeric_label.append(
                object_label(objs[i]) + [0] + object_label(objs[i + 1]) + [1] + [relations_to_idx[rels[i]]])
        return numeric_label

    def decompose_label(numeric_label):
        text_label = []

        for i in range(2):
            shape, size, color, material, pos = numeric_label[i * 5:i * 5 + 5]
            obj = ' '.join([idx_to_sizes[size], idx_to_colors[color],
                            idx_to_materials[material], idx_to_shapes[shape]])
            text_label.append(obj.strip())

        relation = idx_to_relations[numeric_label[-1]]

        # single object
        if relation == 'none':
            return text_label[0]
        else:
            return f'{text_label[0]} {random.choice(description[relation])} {text_label[1]}'

    # figure 1
    objects = [['large', 'gray', 'metal', 'sphere'],
               ['small', 'red', 'metal', 'cube'],
               ['large', 'brown', 'metal', 'cube'],
               ['large', 'green', 'rubber', 'cylinder']]

    relations = ['left', 'right', 'above', 'below', 'front', 'behind']

    relations_permutations = list(permutations(relations, len(objects) - 1))
    object_permutations = list(permutations(objects))
    object_permutations = object_permutations[:1]

    possible_labels = []

    for objects in object_permutations:
        for relations in relations_permutations:
            caption = construct_label(objects, relations)
            possible_labels.append(caption)

    model = load_ebm(ebm_checkpoint_path=ebm_checkpoint_path)

    im = Image.open(image_path).convert('RGB')
    im = im.resize((128, 128), Image.ANTIALIAS)
    im = np.array(im) / 255.
    im = torch.from_numpy(im)

    image_input = im.permute(2, 0, 1)[None].float().to(device)
    labels_input = torch.tensor(possible_labels, dtype=torch.long).to(device)

    label_energies = []

    for i in range(0, len(possible_labels)):
        label = labels_input[i]
        energy = 0
        for j in range(label.shape[0]):
            energy = energy + model(image_input, label[j].unsqueeze(dim=0))
        label_energies.append(energy.sum().item())

    energies = -torch.tensor(label_energies)
    values, indices = energies.topk(1)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        for j in range(len(objects) - 1):
            print(decompose_label(possible_labels[index][j]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--dataset", type=str, default="clevr")
    parser.add_argument("--generalization", action="store_true")

    args = parser.parse_args()
    rank_ebm_captions(args.image_path, args.checkpoint_path)
