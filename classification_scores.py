import os
import json
import argparse

import torch
import numpy as np


from PIL import Image
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from models import Classifier


class ClassificationDataset(Dataset):
	def __init__(self, data_folder, dataset, image_size, num_rels, mode, folder):
		self.image_size = image_size
		if mode == 'generation':
			self.val_path = os.path.join(data_folder, f'{dataset}_generation_{num_rels}_relations.npz')
		elif mode == 'editing':
			self.val_path = os.path.join(data_folder, f'{dataset}_editing_{num_rels}_relations.npz')
		else:
			raise ValueError(f'{dataset} is an invalid dataset type!')

		EXTS = ['jpg', 'jpeg', 'png']
		self.paths = sorted([p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')])

		self.description = {
			"left": "to the left of",
			"right": "to the right of",
			"behind": "behind",
			"front": "in front of",
			"above": "above",
			"below": "below"
		}

		if dataset in ['igibson', 'clevr']:
			with open('./data/attributes.json', 'r') as f:
				data_json = json.load(f)
				self.colors_to_idx = data_json[dataset]['colors']
				self.shapes_to_idx = data_json[dataset]['shapes']
				self.materials_to_idx = data_json[dataset]['materials']
				self.sizes_to_idx = data_json[dataset]['sizes']
				self.relations_to_idx = data_json[dataset]['relations']

				self.idx_to_colors = list(data_json[dataset]['colors'].keys())
				self.idx_to_shapes = list(data_json[dataset]['shapes'].keys())
				self.idx_to_materials = list(data_json[dataset]['materials'].keys())
				self.idx_to_sizes = list(data_json[dataset]['sizes'].keys())
				self.idx_to_relations = list(data_json[dataset]['relations'].keys())
		elif dataset == 'blocks':
			relations = {'below': 0, 'above': 1}
			selected_objects = ['red', 'green', 'blue', 'yellow']
			objects = {object_name: i for i, object_name in enumerate(selected_objects)}
			self.objects = {value: key for key, value in objects.items()}
			self.relations = {value: key for key, value in relations.items()}
		else:
			raise ValueError(f'{dataset} is invalid!')

		# load data
		data = np.load(self.val_path)
		self.labels = data['labels']
		self.ims = data['ims']

		data_info = {
			'dataset': dataset,
			'dataset size': self.__len__(),
			'number of relations': num_rels,
			'data path': self.val_path
		}

		for key, value in data_info.items():
			print(f'{key}: {value}')

	def __getitem__(self, index):
		path = self.paths[index]
		im = Image.open(path)
		im = im.resize((self.image_size, self.image_size), Image.ANTIALIAS)
		im = np.array(im) / 255.
		im = torch.from_numpy(im)
		label = torch.from_numpy(self.labels[index])    # gt label
		return im, label

	def __len__(self):
		return len(self.paths)


def load_classifier(checkpoint_dir, dataset):
	folder = os.path.join(checkpoint_dir, f'{dataset}_classifier')
	paths = sorted([
		int(str(p).split('/')[-1].replace('.tar', ''))
		for ext in ['tar'] for p in Path(f'{folder}').glob(f'**/*.{ext}')
	])
	latest_checkpoint_path = os.path.join(folder, f'{paths[-1]}.tar')
	checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
	print(f'loading from the latest checkpoint: {latest_checkpoint_path} with val acc: {checkpoint["val"]}')
	model = Classifier(checkpoint['args'])
	model.load_state_dict(checkpoint['model_state_dict'])
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	return model.eval()


def compute_classification_score(classifier, data_folder, dataset, generated_img_folder, image_size, num_rels, mode):
	dataset = ClassificationDataset(
		data_folder=data_folder, dataset=dataset, image_size=image_size,
		folder=generated_img_folder, num_rels=num_rels, mode=mode
	)

	dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=32, drop_last=False, num_workers=4, pin_memory=True)

	total_corrects, total_ims = 0, 0
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	for i, (gen_ims, gt_labels) in enumerate(tqdm(dataloader)):
		gen_ims = gen_ims.permute((0, 3, 1, 2)).float().to(device)
		gt_labels = gt_labels.to(device)

		if len(gt_labels.shape) == 3:
			labels = torch.chunk(gt_labels, chunks=gt_labels.shape[1], dim=1)
			labels = [chunk.squeeze(dim=1) for chunk in labels]
		else:
			labels = [gt_labels]

		result = torch.zeros((gen_ims.shape[0], 1), dtype=torch.long, device=device)
		for label in labels:
			with torch.no_grad():
				outputs = classifier(gen_ims, label)
				result += torch.round(outputs).long()

		corrects = torch.sum(result == len(labels))

		total_corrects += corrects.item()
		total_ims += gen_ims.shape[0]

	print(f'{generated_img_folder} has a classification scores: ', total_corrects / total_ims)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# classifier flag
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--dataset", choices=['clevr', 'igibson'])
	parser.add_argument("--checkpoint_dir", type=str)

	# input images
	parser.add_argument("--im_size", default=128)
	parser.add_argument("--data_folder", type=str)
	parser.add_argument("--generated_img_folder", type=str)
	parser.add_argument("--num_rels", type=int)
	parser.add_argument("--mode", choices=['generation', 'editing'])

	args = parser.parse_args()

	classifier = load_classifier(checkpoint_dir=args.checkpoint_dir, dataset=args.dataset)

	compute_classification_score(
		classifier=classifier, data_folder=args.data_folder, dataset=args.dataset,
		generated_img_folder=args.generated_img_folder, image_size=args.im_size,
		num_rels=args.num_rels, mode=args.mode
	)
