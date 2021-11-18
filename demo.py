import os
import json
import argparse
import random

import imageio

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from skimage import img_as_ubyte

from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader

from models import ResNetModel, ResNetCLIP


class InferenceDataset(Dataset):
	def __init__(self, data_folder, dataset, image_size, num_rels, mode, clip=False, clip_all=False, invert_rel=False):
		self.invert_rel = invert_rel
		self.image_size = image_size
		if mode == 'generation':
			self.val_path = os.path.join(data_folder, f'{dataset}_generation_{num_rels}_relations.npz')
		elif mode == 'editing':
			self.val_path = os.path.join(data_folder, f'{dataset}_editing_{num_rels}_relations.npz')
		else:
			raise ValueError(f'{mode} is an invalid mode type!')

		# load precompute clip features if is_clip is enabled
		if clip:
			if clip_all:
				clip_feature_path = os.path.join(data_folder, f'clip_all_features_{dataset}.pt')
				self.clip_features = torch.load(clip_feature_path, map_location='cpu')
			else:
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
		self.labels = data['labels']

		data_info = {
			'dataset': dataset,
			'dataset size': self.__len__(),
			'number of relations': num_rels,
			'data path': self.val_path
		}

		for key, value in data_info.items():
			print(f'{key}: {value}')

	def __getitem__(self, index):
		im = Image.fromarray(self.ims[index])
		im = im.resize((self.image_size, self.image_size), Image.ANTIALIAS)
		im = np.array(im) / 255.
		im = torch.from_numpy(im)
		label = torch.from_numpy(self.labels[index])
		if self.invert_rel:
			label = self._invert_relation(label)
		return im, label, self.get_caption(label)

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
	def get_caption(self, label):
		# decompose label into multiple single object relation
		label = torch.chunk(label, chunks=label.shape[0], dim=0)
		label = [y.squeeze() for y in label]
		caption = '\n'.join([self._label_to_caption(y) for y in label])
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

	def _invert_relation(self, label):
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
			label[i][-1] = self.relations_to_idx[new_relation]
		return label

	def clip_collate_fn(self, batches):
		ims, labels, captions = zip(*batches)
		batch_ims = torch.stack(ims, dim=0)
		batch_labels = self._encode_batch_labels(torch.stack(labels, dim=0))
		return batch_ims, batch_labels, captions

	def clip_all_collate_fn(self, batches):
		ims, labels, captions = zip(*batches)
		batch_ims = torch.stack(ims, dim=0)
		batch_labels = []
		for caption in captions:
			clip_all_features = []
			num_rel_captions = caption.split('\n')
			for sub_cap in num_rel_captions:
				clip_all_features.append(self.clip_features[sub_cap])
			batch_labels.append(np.concatenate(clip_all_features, axis=0))
		batch_labels = torch.from_numpy(np.array(batch_labels))
		return batch_ims, batch_labels, captions


def get_color_distortion(s=1.0):
	color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.4 * s)
	rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
	rnd_gray = transforms.RandomGrayscale(p=0.2)
	color_distort = transforms.Compose([
		rnd_color_jitter,
		rnd_gray]
	)
	return color_distort


def gen_images(model, dataset, labels, num_steps, step_lr, im_size, batch_size, clip, clip_all, device):
	torch.seed()
	# sampling augmentation
	if dataset in ['visual_genome', 'blocks']:
		transform = transforms.Compose([
			transforms.RandomResizedCrop(im_size, scale=(0.8, 1.0)),
			get_color_distortion(0.1),
			transforms.ToTensor()]
		)
	else:
		transform = transforms.Compose([
			transforms.RandomResizedCrop(im_size, scale=(0.08, 1.0)),
			get_color_distortion(0.5),
			transforms.ToTensor()]
		)

	im = torch.rand(batch_size, 3, im_size, im_size).to(device)
	im_noise = torch.randn_like(im).detach()

	# a list of CLIP label Mx5 where M is the number fo relations
	if clip and not clip_all:
		for i in range(len(labels)):
			for j in range(len(labels[i])):
				labels[i][j] = labels[i][j].to(device)
	else:
		if len(labels.shape) == 2:  # Nx11 --> Nx1x11
			labels = labels[:, None]

		labels = labels.to(device)
		labels = torch.chunk(labels, chunks=labels.shape[1], dim=1)  # NxMx11 --> [Nx11] * M
		labels = [chunk.squeeze(dim=1) for chunk in labels]

	# scale the step size by the number of labels we compose
	step_lr /= len(labels)
	init_num_data_aug = 10  # tunable
	init_num_ld = 20  # tunable

	grid = make_grid(im, normalize=True, nrow=int(batch_size ** 0.5)).detach().cpu().permute((1, 2, 0)).numpy()
	videos = [(grid * 255.).astype(np.uint8)]  # GIF

	for i in range(init_num_data_aug):
		for j in range(init_num_ld):
			im_noise.normal_()
			im = im + 0.001 * im_noise
			im.requires_grad_(requires_grad=True)

			energy = sum([model.forward(im, y) for y in labels])

			im_grad = torch.autograd.grad([energy.sum()], [im])[0]

			im = im - step_lr * im_grad
			im = im.detach()
			im = torch.clamp(im, 0, 1)

			grid = make_grid(im, nrow=int(batch_size ** 0.5)).detach().cpu().permute((1, 2, 0)).numpy()
			videos.append((grid * 255.).astype(np.uint8))

	# Langevin dynamic sampling - tunable (refinement)
	for i in range(num_steps):
		im_noise.normal_()
		im = im + 0.001 * im_noise
		im.requires_grad_(requires_grad=True)
		energy = sum([model.forward(im, y) for y in labels])
		print('step', i, 'energy', energy.mean())
		im_grad = torch.autograd.grad([energy.sum()], [im])[0]
		im = im - step_lr * im_grad
		im = im.detach()
		im = torch.clamp(im, 0, 1)

		grid = make_grid(im, nrow=int(batch_size ** 0.5)).detach().cpu().permute((1, 2, 0)).numpy()
		videos.append((grid * 255.).astype(np.uint8))

	return im, videos


def edit_images(im, model, labels, num_steps, step_lr, clip, clip_all, device):
	torch.seed()
	im = im.permute((0, 3, 1, 2)).float().to(device)
	im_noise = torch.randn_like(im).detach()

	if clip and not clip_all:
		for i in range(len(labels)):
			for j in range(len(labels[i])):
				labels[i][j] = labels[i][j].to(device)
	else:
		labels = labels.to(device)
		labels = torch.chunk(labels, chunks=labels.shape[1], dim=1)  # NxMx11 --> [Nx11] * M
		labels = [chunk.squeeze(dim=1) for chunk in labels]

	step_lr /= len(labels)

	grid = make_grid(im, nrow=int(im.shape[0] ** 0.5)).detach().cpu().permute((1, 2, 0)).numpy()
	videos = [(grid * 255.).astype(np.uint8)]  # GIF

	for i in range(num_steps):
		im_noise.normal_()
		im = im + 0.001 * im_noise
		im.requires_grad_(requires_grad=True)

		energy = sum([model.forward(im, y) for y in labels])
		im_grad = torch.autograd.grad([energy.sum()], [im])[0]
		print(i, energy.mean())
		im = im - step_lr * im_grad
		im = im.detach()
		im = torch.clamp(im, 0, 1)

		grid = make_grid(im, nrow=int(im.shape[0] ** 0.5)).detach().cpu().permute((1, 2, 0)).numpy()
		videos.append((grid * 255.).astype(np.uint8))

	return im, videos


def inference_example(
		checkpoint_folder: str,
		model_name: str,
		resume_iter: int,
		data_folder: str,
		batch_size: int,
		num_rels: int,
		output_folder: str,
		clip: bool,
		clip_all: bool,
		mode: str,
		invert_rel: bool,
		num_steps: int = 80,  # the number of Langevin Sampling steps for the second phase
):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_path = os.path.join(checkpoint_folder, model_name, f"model_{resume_iter}.pth")
	checkpoint = torch.load(model_path, map_location='cpu')
	FLAGS = checkpoint['FLAGS']

	# load model
	if clip or clip_all:
		model = ResNetCLIP(FLAGS)
	else:
		model = ResNetModel(FLAGS)

	model.load_state_dict(checkpoint['model_state_dict_0'])
	model = model.eval().to(device)

	val_dataset = InferenceDataset(
		data_folder=data_folder,
		dataset=FLAGS.dataset,
		image_size=FLAGS.im_size,
		num_rels=num_rels,
		clip=clip,
		mode=mode,
		invert_rel=invert_rel
	)

	collate_function = None
	if clip:
		if clip_all:
			collate_function = val_dataset.clip_all_collate_fn
		else:
			collate_function = val_dataset.clip_collate_fn

	# shuffle = False --> make sure each image is generated based on label's ordering
	dataloader = DataLoader(
		dataset=val_dataset, shuffle=False, drop_last=False, batch_size=batch_size, collate_fn=collate_function
	)

	# create output folder
	image_output_path = os.path.join(output_folder, model_name, f'num_rel_{num_rels}')
	os.makedirs(image_output_path, exist_ok=True)

	ims, labels, captions = next(iter(dataloader))

	if mode == 'generation':
		results, videos = gen_images(
			model=model, dataset=FLAGS.dataset, labels=labels, num_steps=num_steps, clip=clip,
			clip_all=clip_all, step_lr=FLAGS.step_lr, im_size=FLAGS.im_size, batch_size=ims.shape[0], device=device
		)
	elif mode == 'editing':
		results, videos = edit_images(
			im=ims, model=model, labels=labels, num_steps=num_steps,
			clip=clip, clip_all=clip_all, step_lr=FLAGS.step_lr, device=device
		)
	else:
		raise ValueError(f'{mode} is an invalid mode!')

	# save each image
	original_grid = make_grid(ims.permute((0, 3, 1, 2)).detach().cpu(), nrow=int(ims.shape[0] ** 0.5))
	save_image(original_grid, fp=f'./samples/original_{mode}_samples.png')
	grid = make_grid(results.detach().cpu(), nrow=int(results.shape[0] ** 0.5))
	save_image(grid, fp=f'./samples/{mode}_samples.png')
	# save GIF
	imageio.mimsave(f'./samples/{mode}_samples.gif', videos)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_folder", type=str, required=True)
	parser.add_argument("--model_name", type=str, required=True)
	parser.add_argument("--output_folder", type=str, required=True)
	parser.add_argument("--data_folder", type=str, required=True)
	parser.add_argument("--dataset", choices=['clevr', 'igibson', 'visual_genome', 'blocks'], required=True)
	parser.add_argument("--resume_iter", type=str, required=True)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_steps", type=int, default=80)
	parser.add_argument("--num_rels", type=int, required=True)
	parser.add_argument("--clip", action='store_true')
	parser.add_argument("--clip_all", action="store_true")
	parser.add_argument("--mode", choices=['generation', 'editing'])

	# editing argument
	parser.add_argument("--invert_rel", action="store_true")
	args = parser.parse_args()

	inference_example(
		checkpoint_folder=args.checkpoint_folder,
		model_name=args.model_name,
		resume_iter=args.resume_iter,
		data_folder=args.data_folder,
		batch_size=args.batch_size,
		num_rels=args.num_rels,
		output_folder=args.output_folder,
		num_steps=args.num_steps,
		mode=args.mode,
		clip=args.clip,
		invert_rel=args.invert_rel,
		clip_all=args.clip_all
	)
