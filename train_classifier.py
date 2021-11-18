import json
import os
import time
import copy
import argparse

import numpy as np

import torch
import torch.optim as optim

from tqdm import tqdm
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from models import Classifier


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ObjectRelationsDataset(Dataset):
	def __init__(self, data_path, image_size, dataset, split, cond=False, sample_neg=False):
		super().__init__()
		data = np.load(data_path)
		self.cond = cond
		self.ims = data['ims']
		self.labels = data['labels']
		self.image_size = image_size
		self.sample_neg = sample_neg

		if split == 'train':
			end = self.ims.shape[0] * 4 // 5
			self.ims = self.ims[:end]
			self.labels = self.labels[:end]
		elif split == 'val':
			start = self.ims.shape[0] * 4 // 5
			self.ims = self.ims[start:]
			self.labels = self.labels[start:]
		else:
			raise NotImplementedError

		self.transform = transforms.Compose([
			transforms.ToPILImage(mode='RGB'),
			transforms.Resize(image_size),
			transforms.CenterCrop(image_size),
			transforms.ToTensor(),
		])

		self._load_attributes(dataset)
		print(f'loading dataset from {data_path}...')

	def __len__(self):
		return self.ims.shape[0]

	def __getitem__(self, index):
		if self.sample_neg:
			neg_index = np.random.randint(0, self.__len__())
			while neg_index == index and np.array_equal(self.labels[index], self.labels[neg_index]):
				neg_index = np.random.randint(0, self.__len__())
			return self.transform(self.ims[index]), self.labels[index], self.transform(self.ims[neg_index])
		else:
			return self.transform(self.ims[index]), self.labels[index]

	def _load_attributes(self, dataset):
		self.description = {
			"left": "to the left of",
			"right": "to the right of",
			"behind": "behind",
			"front": "in front of",
			"above": "above",
			"below": "below"
		}

		with open('./data/attributes.json', 'r') as f:
			data_json = json.load(f)
			self.colors_to_idx = data_json[dataset]['colors']
			self.shapes_to_idx = data_json[dataset]['shapes']
			self.materials_to_idx = data_json[dataset]['materials']
			self.sizes_to_idx = data_json[dataset]['sizes']
			self.relations_to_idx = data_json[dataset]['relations']

			self.idx_to_color = list(data_json[dataset]['colors'].keys())
			self.idx_to_shape = list(data_json[dataset]['shapes'].keys())
			self.idx_to_material = list(data_json[dataset]['materials'].keys())
			self.idx_to_size = list(data_json[dataset]['sizes'].keys())
			self.idx_to_relation = list(data_json[dataset]['relations'].keys())


def train_binary_classification(args):
	if args.pretrained:
		folder = os.path.join(args.checkpoint_dir, f'{args.dataset}_classifier')
		paths = sorted([
			int(str(p).split('/')[-1].replace('.tar', ''))
			for ext in ['tar'] for p in Path(f'{folder}').glob(f'**/*.{ext}')
		])
		latest_checkpoint_path = os.path.join(folder, f'{paths[-1]}.tar')
		checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
		model = Classifier(checkpoint['args'])
		print(checkpoint['val'])
	else:
		model = Classifier(args)

	model = model.train().to(device)
	if args.dataset in ['clevr', 'igibson']:
		optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
	elif args.dataset == 'blocks':
		optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
	else:
		raise NotImplementedError

	if args.dataset in ['clevr', 'igibson']:
		datasets = {phase: ObjectRelationsDataset(
			data_path=args.data_path,
			image_size=128, dataset=args.dataset, split=phase, sample_neg=True)
			for phase in ['train', 'val']}
	else:
		raise ValueError()

	dataset_sizes = {phase: len(datasets[phase]) for phase in ['train', 'val']}
	dataloaders = {phase: DataLoader(
		dataset=datasets[phase], shuffle=True, pin_memory=True, num_workers=4, batch_size=args.batch_size)
		for phase in ['train', 'val']
	}

	criterion = torch.nn.BCELoss()
	checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.dataset}_classifier')
	os.makedirs(checkpoint_path, exist_ok=True)
	train_model(model, dataloaders, criterion, optimizer, dataset_sizes, checkpoint_path, 0, 50)


def train_model(model, dataloaders, criterion, optimizer, dataset_sizes, checkpoint_path, start_epoch=0, num_epochs=50):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_epoch = 0
	best_acc = 0.0

	early_stopping_patience = 5

	for epoch in range(start_epoch, num_epochs):
		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for i, (inputs, labels, neg_inputs) in enumerate(tqdm(dataloaders[phase])):
				inputs = inputs.float().to(device)
				neg_inputs = neg_inputs.float().to(device)

				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					pos_logits = model(inputs, labels)
					neg_logits = model(neg_inputs, labels)

					loss = criterion(pos_logits, torch.ones_like(pos_logits)) + criterion(neg_logits, torch.zeros_like(neg_logits))

					if phase == 'train':
						loss.backward()
						optimizer.step()

					current_loss = loss.item() * inputs.size(0)

					pos_preds = torch.round(pos_logits)
					neg_preds = torch.round(neg_logits)

					pos_corrects = torch.sum(pos_preds == 1)
					neg_correcst = torch.sum(neg_preds == 0)

					corrects = pos_corrects + neg_correcst

					print('loss', loss.item(), 'acc', corrects.item() / 2 / inputs.shape[0])

					running_loss += current_loss
					running_corrects += corrects.item()

			epoch_loss = running_loss / (2 * dataset_sizes[phase])
			epoch_acc = running_corrects / (2 * dataset_sizes[phase])

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_epoch = epoch
				best_model_wts = copy.deepcopy(model.state_dict())

				torch.save({
					'epoch': epoch + 1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': epoch_loss,
					'val': epoch_acc,
					'args': args
				},
					os.path.join(checkpoint_path, f'{epoch + 1}.tar')
				)

		# check early stopping criterion
		if epoch + 1 - best_epoch > early_stopping_patience:
			print(f'Early Stopping at epoch: {epoch + 1}')
			break

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# models
	parser.add_argument("--train", action="store_true")
	parser.add_argument("--pretrained", action="store_true")
	parser.add_argument("--spec_norm", action="store_true")
	parser.add_argument("--norm", action="store_true")
	parser.add_argument("--alias", action="store_true")

	parser.add_argument("--filter_dim", type=int, default=64)
	parser.add_argument("--im_size", type=int, default=128)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--batch_size", type=int, default=32)

	parser.add_argument("--dataset", choices=['clevr', 'igibson'])

	parser.add_argument("--checkpoint_dir", type=str, default='./')

	args = parser.parse_args()

	if args.train:
		train_binary_classification(args)
