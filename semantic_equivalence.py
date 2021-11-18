import os
import json
import argparse

import clip
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset, DataLoader

from models import ResNetModel, ResNetCLIP, Classifier


class SemanticEquivalenceDataset(Dataset):
	def __init__(self, data_folder, dataset, image_size, num_rels, clip=False, transform=None):
		self.image_size = image_size
		self.val_path = os.path.join(data_folder, f'clevr_semantic_equivalence.npz')
		# load precompute clip features if is_clip is enabled
		if clip:
			clip_feature_path = os.path.join(data_folder, f'clip_features_{dataset}.pt')
			self.clip_features = torch.load(clip_feature_path, map_location='cpu')

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
		self.ims = data['ims']
		self.labels = data['labels'][:, :num_rels]

		data_info = {
			'dataset': dataset,
			'dataset size': self.__len__(),
			'number of relations': num_rels,
			'data path': self.val_path
		}

		for key, value in data_info.items():
			print(f'{key}: {value}')

		self.transform = transform

	def __getitem__(self, index):
		im = Image.fromarray(self.ims[index])

		im = im.resize((self.image_size, self.image_size), Image.ANTIALIAS)
		im = np.array(im) / 255.
		im = torch.from_numpy(im)

		label = torch.from_numpy(self.labels[index])    # gt label
		return im, label, self._extract_equivalent_pair(label), self._extract_mismatching_label(label)

	def __len__(self):
		return self.ims.shape[0]

	def _get_object_description(self, object_label):
		shape, size, color, material, _ = object_label
		object_des = ' '.join([
			self.idx_to_sizes[size], self.idx_to_colors[color],
			self.idx_to_materials[material], self.idx_to_shapes[shape]
		])
		return object_des.strip()

	def _label_to_caption(self, label):
		obj_des_1 = self._get_object_description(label[:5])
		obj_des_2 = self._get_object_description(label[5:10])
		relation = self.idx_to_relations[label[-1]]
		if relation == 'none':  # single object
			return obj_des_1
		else:
			return ' '.join([obj_des_1, self.description[relation], obj_des_2]).strip()

	# helper function for extracting the label in text form
	def get_caption(self, label, delimiter='and'):
		# decompose label into multiple single object relation
		label = torch.chunk(label, chunks=label.shape[0], dim=0)
		label = [y.squeeze() for y in label]
		caption = [self._label_to_caption(y).strip() for y in label]
		return caption

	def _clip_encoded_label(self, label):
		# extract object text descriptions
		obj_des_1 = self._get_object_description(label[:5])
		obj_des_2 = self._get_object_description(label[5:10])
		# encode them into CLIP embedding
		obj_des_embed_1 = self.clip_features[obj_des_1]
		obj_des_embed_2 = self.clip_features[obj_des_2]
		# other info
		# range [0, 2] where 0, 1 indicate 1st and 2nd object
		# 2 indicates dummy object so (single object image) doesn't have second object
		obj_idx_1, obj_idx_2 = label[4], label[9]
		relation_idx = label[-1]
		return obj_des_embed_1, obj_des_embed_2, obj_idx_1, obj_idx_2, relation_idx

	def _encode_batch_labels(self, labels):
		"""

		Args:
			labels: labels where labels has a shape of BxMxK where M is the number of relations

		Returns:
			a list of input embedding and the size depends on the number of relations we want to compose

		"""
		num_relations = labels.shape[1]  # BxMxK
		encoded_labels = []

		for i in range(num_relations):
			batch_obj_emb_1, batch_obj_emb_2 = [], []
			batch_obj_idx_1, batch_obj_idx_2 = [], []
			batch_rel_idx = []

			for j in range(labels.shape[0]):
				obj_des_embed_1, obj_des_embed_2, obj_idx_1, obj_idx_2, relation_idx = self._clip_encoded_label(
					labels[j][i])
				batch_obj_emb_1.append(obj_des_embed_1)
				batch_obj_emb_2.append(obj_des_embed_2)
				batch_obj_idx_1.append(obj_idx_1)
				batch_obj_idx_2.append(obj_idx_2)
				batch_rel_idx.append(relation_idx)

			# convert to tensors
			batch_obj_emb_1 = torch.cat(batch_obj_emb_1, dim=0)
			batch_obj_emb_2 = torch.cat(batch_obj_emb_2, dim=0)
			batch_obj_idx_1 = torch.tensor(batch_obj_idx_1, dtype=torch.long)
			batch_obj_idx_2 = torch.tensor(batch_obj_idx_2, dtype=torch.long)
			batch_rel_idx = torch.tensor(batch_rel_idx, dtype=torch.long)

			encoded_labels.append([batch_obj_emb_1, batch_obj_emb_2, batch_obj_idx_1, batch_obj_idx_2, batch_rel_idx])

		return encoded_labels

	def _inverse_relation(self, label):
		new_label = torch.clone(label)
		for i in range(label.shape[0]):
			relation = self.idx_to_relations[label[i][-1]]
			# change the relation such that 'left' -> 'right', 'above' -> 'below
			new_relation = {
				'left': 'right',
				'right': 'left',
				'front': 'behind',
				'behind': 'front',
				'above': 'below',
				'below': 'above'
			}.get(relation)
			new_label[i][-1] = self.relations_to_idx[new_relation]
		return new_label

	def _extract_mismatching_label(self, label):
		return self._inverse_relation(label)

	def _extract_equivalent_pair(self, label):
		# inverse_relation
		inverse_label = self._inverse_relation(label)
		# swap object attribute positions
		new_label = torch.cat((
			inverse_label[:, 5:9], inverse_label[:, 4:5],
			inverse_label[:, 0:4], inverse_label[:, 9:10], inverse_label[:, 10:]), dim=1)
		return new_label

	def clip_collate_fn(self, batches):
		ims, labels, eq_labels, mis_labels = zip(*batches)
		batch_ims = torch.stack(ims, dim=0)
		batch_labels = self._encode_batch_labels(torch.stack(labels, dim=0))
		batch_eq_labels = self._encode_batch_labels(torch.stack(eq_labels, dim=0))
		batch_mis_labels = self._encode_batch_labels(torch.stack(mis_labels, dim=0))
		return batch_ims, batch_labels, batch_eq_labels, batch_mis_labels

	def caption_collate_fn(self, batches):
		ims, labels, eq_labels, mis_labels = zip(*batches)
		batch_ims = torch.stack(ims, dim=0)
		batch_captions = [self.get_caption(labels[i]) for i in range(batch_ims.shape[0])]
		batch_eq_captions = [self.get_caption(eq_labels[i]) for i in range(batch_ims.shape[0])]
		batch_mis_captions = [self.get_caption(mis_labels[i]) for i in range(batch_ims.shape[0])]
		return batch_ims, batch_captions, batch_eq_captions, batch_mis_captions


def ebm_semantic_equivalence_score(
		data_folder,
		num_rels,
		batch_size,
		checkpoint_folder,
		model_name,
		resume_iter,
		clip
):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_path = os.path.join(checkpoint_folder, model_name, f"model_{resume_iter}.pth")
	checkpoint = torch.load(model_path, map_location='cpu')
	FLAGS = checkpoint['FLAGS']

	# load model
	if clip:
		model = ResNetCLIP(FLAGS, FLAGS.relation)
	else:
		model = ResNetModel(FLAGS, FLAGS.relation)

	model.load_state_dict(checkpoint['model_state_dict_0'])
	model = model.eval().to(device)

	val_dataset = SemanticEquivalenceDataset(
		data_folder=data_folder,
		dataset=FLAGS.dataset,
		image_size=FLAGS.im_size,
		num_rels=num_rels,
		clip=clip,
	)

	# shuffle = False --> make sure each image is generated based on label's ordering
	dataloader = DataLoader(
		dataset=val_dataset, shuffle=False, drop_last=False, batch_size=batch_size,
		collate_fn=val_dataset.clip_collate_fn if clip else None
	)

	corrects, total_samples = 0, 0

	# file = open('caption.txt', 'w')

	for i, (ims, labels, eq_labels, mis_labels) in enumerate(tqdm(dataloader)):
		ims = ims.permute((0, 3, 1, 2)).float().to(device)

		split_labels = []
		for curr_label in [labels, eq_labels, mis_labels]:
			if clip:
				for i in range(len(curr_label)):
					for j in range(len(curr_label[i])):
						curr_label[i][j] = curr_label[i][j].to(device)
			else:
				curr_label = curr_label.to(device)
				curr_label = torch.chunk(curr_label, chunks=curr_label.shape[1], dim=1)  # NxMx11 --> [Nx11] * M
				curr_label = [chunk.squeeze(dim=1) for chunk in curr_label]

			split_labels.append(curr_label)

		pred = torch.zeros((ims.shape[0]), dtype=torch.long, device=device)
		with torch.no_grad():
			for i in range(num_rels):
				energies = []
				for curr_label in split_labels:
					energies.append(model(ims, curr_label[i]).sum(dim=1))
				# measure curr relation's relative difference
				energies, eq_energies, mis_energies = energies
				equiv_energy_diff = torch.abs(energies - eq_energies)
				mis_energy_diff = torch.abs(energies - mis_energies)
				pred += (equiv_energy_diff < mis_energy_diff).long()

		corrects += torch.sum(pred == num_rels).item()
		total_samples += ims.shape[0]
		# break

	print(corrects / total_samples)
	return corrects / total_samples


def clip_semantic_equivalence_score(
	data_folder,
	num_rels,
	batch_size,
	dataset,
	checkpoint_path,
):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	clip_model, preprocess = clip.load("ViT-B/32", device=device)

	input_resolution = clip_model.input_resolution
	context_length = clip_model.context_length
	vocab_size = clip_model.vocab_size

	if checkpoint_path is not None:
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		checkpoint['model_state_dict']['input_resolution'] = input_resolution
		checkpoint['model_state_dict']['context_length'] = context_length
		checkpoint['model_state_dict']['vocab_size'] = vocab_size
		clip_model.load_state_dict(checkpoint['model_state_dict'])
		print(f'loading fine-tuned CLIP: {checkpoint_path}')
	else:
		print(f'loading pre-trained CLIP')

	clip_model.eval()

	val_dataset = SemanticEquivalenceDataset(
		data_folder=data_folder,
		dataset=dataset,
		image_size=input_resolution,
		num_rels=num_rels,
		transform=preprocess
	)

	# shuffle = False --> make sure each image is generated based on label's ordering
	dataloader = DataLoader(
		dataset=val_dataset, shuffle=False, drop_last=False,
		batch_size=batch_size, collate_fn=val_dataset.caption_collate_fn
	)
	corrects, total_samples = 0, 0

	for _, (ims, captions, eq_captions, mis_captions) in enumerate(tqdm(dataloader)):
		assert len(captions[0]) == len(eq_captions[0]) == len(mis_captions[0]) == num_rels
		ims = ims.permute((0, 3, 1, 2)).float().to(device)
		results = torch.zeros((ims.shape[0]), dtype=torch.long, device=device)

		for i in range(num_rels):
			i_captions = [captions[j][i] for j in range(len(captions))]
			i_eq_captions = [eq_captions[j][i] for j in range(len(eq_captions))]
			i_mis_captions = [mis_captions[j][i] for j in range(len(mis_captions))]

			with torch.no_grad():
				similarities = []
				for curr_captions in [i_captions, i_eq_captions, i_mis_captions]:
					tokenized_text = torch.cat([clip.tokenize(cap) for cap in curr_captions], dim=0)
					image_features = clip_model.encode_image(ims.to(device))
					text_features = clip_model.encode_text(tokenized_text.to(device))

					image_features /= image_features.norm(dim=-1, keepdim=True)
					text_features /= text_features.norm(dim=-1, keepdim=True)
					similarity = (100.0 * image_features @ text_features.T)
					similarity = torch.diagonal(similarity, 0)
					similarities.append(similarity)

				original_similarity, equiv_similarity, mis_similarity = similarities
				equiv_diff = torch.abs(original_similarity - equiv_similarity)
				mis_diff = torch.abs(equiv_similarity - mis_similarity)
				results += (equiv_diff < mis_diff)

		corrects += torch.sum(results == num_rels).item()
		total_samples += ims.shape[0]

	print(corrects / total_samples)
	return corrects / total_samples


def classifier_semantic_equivalence_score(
		data_folder,
		num_rels,
		batch_size,
		dataset,
		im_size,
		checkpoint_path,
):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
		model = model.to(device)
		return model.eval()

	model = load_classifier(checkpoint_dir=checkpoint_path, dataset=dataset)

	val_dataset = SemanticEquivalenceDataset(
		data_folder=data_folder,
		dataset=dataset,
		image_size=im_size,
		num_rels=num_rels,
	)

	# shuffle = False --> make sure each image is generated based on label's ordering
	dataloader = DataLoader(dataset=val_dataset, shuffle=False, drop_last=False, batch_size=batch_size)

	corrects, total_samples = 0, 0

	for _, (ims, labels, eq_labels, mis_labels) in enumerate(tqdm(dataloader)):
		ims = ims.permute((0, 3, 1, 2)).float().to(device)
		split_labels = []
		for curr_label in [labels, eq_labels, mis_labels]:
			curr_label = curr_label.to(device)
			curr_label = torch.chunk(curr_label, chunks=curr_label.shape[1], dim=1)  # NxMx11 --> [Nx11] * M
			curr_label = [chunk.squeeze(dim=1) for chunk in curr_label]
			split_labels.append(curr_label)

		pred = torch.zeros((ims.shape[0]), dtype=torch.long, device=device)
		with torch.no_grad():
			for i in range(num_rels):
				energies = []
				for curr_label in split_labels:
					assert len(curr_label) == num_rels
					energies.append(model(ims, curr_label[i]).sum(dim=1))

				# measure curr relation's relative difference
				energies, eq_energies, mis_energies = energies
				equiv_energy_diff = torch.abs(energies - eq_energies)
				mis_energy_diff = torch.abs(energies - mis_energies)
				pred += (equiv_energy_diff < mis_energy_diff).long()

		corrects += torch.sum(pred == num_rels).item()
		total_samples += ims.shape[0]

	print(corrects / total_samples)
	return corrects / total_samples


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", choices=['clip', 'ebm-emb', 'ebm-clip', 'classifier'])
	# model setting (EBMs)
	parser.add_argument("--checkpoint_folder", type=str)
	parser.add_argument("--model_name", type=str)
	parser.add_argument("--resume_iter", type=str)
	parser.add_argument("--clip", action='store_true')

	# Classifier setting
	parser.add_argument("--classifier_folder", type=str)

	# CLIP setting
	parser.add_argument("--fine_tune_clip_path", type=str)

	parser.add_argument("--data_folder", type=str, required=True)
	parser.add_argument("--dataset", choices=['clevr', 'igibson'], required=True)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_rels", type=int, required=True)

	args = parser.parse_args()

	if args.model == 'clip':
		clip_semantic_equivalence_score(
			data_folder=args.data_folder, num_rels=args.num_rels, batch_size=args.batch_size,
			dataset=args.dataset, checkpoint_path=args.fine_tune_clip_path
		)
	elif args.model == 'ebm-emb':
		ebm_semantic_equivalence_score(
			data_folder=args.data_folder,
			num_rels=args.num_rels,
			batch_size=args.batch_size,
			checkpoint_folder=args.checkpoint_folder,
			model_name=args.model_name,
			resume_iter=args.resume_iter,
			clip=False
		)
	elif args.model == 'ebm-clip':
		ebm_semantic_equivalence_score(
			data_folder=args.data_folder,
			num_rels=args.num_rels,
			batch_size=args.batch_size,
			checkpoint_folder=args.checkpoint_folder,
			model_name=args.model_name,
			resume_iter=args.resume_iter,
			clip=True
		)
	elif args.model == 'classifier':
		classifier_semantic_equivalence_score(
			data_folder=args.data_folder,
			num_rels=args.num_rels,
			batch_size=args.batch_size,
			dataset=args.dataset,
			im_size=128,
			checkpoint_path=args.checkpoint_folder
		)
