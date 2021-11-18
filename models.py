import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.utils import swish, CondResBlock, Downsample, Self_Attn


class ResNetModel(nn.Module):
	def __init__(self, args):
		super(ResNetModel, self).__init__()
		self.act = swish

		self.args = args
		self.spec_norm = args.spec_norm
		self.norm = args.norm
		self.dataset = args.dataset

		if args.dataset == 'clevr':
			self.index_embedding = nn.Embedding(3, 64)
			self.shape_embedding = nn.Embedding(4, 64)
			self.color_embedding = nn.Embedding(9, 64)
			self.size_embedding = nn.Embedding(3, 64)
			self.material_embedding = nn.Embedding(3, 64)
			self.relation_embedding = nn.Embedding(7, 512)

			self.fc = nn.Linear(320, 512)
			self.res_classes = 1536
		elif args.dataset == 'igibson':
			self.index_embedding = nn.Embedding(3, 64)
			self.shape_embedding = nn.Embedding(5, 64)
			self.color_embedding = nn.Embedding(9, 64)  # previous 6, now 9
			self.size_embedding = nn.Embedding(2, 64)
			self.material_embedding = nn.Embedding(5, 64)  # previous 4, now 5
			self.relation_embedding = nn.Embedding(7, 512)

			self.fc = nn.Linear(320, 512)
			self.res_classes = 1536
		elif args.dataset == 'visual_genome':
			self.obj_embedding = nn.Embedding(31, 4)  # 31 objects
			self.relation_embedding = nn.Embedding(11, 4)  # 11 relations
			# 4*3 + 2 indices embedding
			# self.res_classes = 14 # previous
			self.fc = nn.Linear(14, 16)  # now
			self.res_classes = 16
		elif args.dataset == 'blocks':
			self.obj_embedding = nn.Embedding(4, 16)  # 31 objects
			self.relation_embedding = nn.Embedding(2, 16)  # 11 relations
			# 3 * 16 + 2 indices embedding
			self.res_classes = 50
		else:
			raise NotImplementedError

		self.input_channel = 3
		self.downsample = Downsample(channels=self.input_channel)
		self.cond = args.cond
		self.batch_size = args.batch_size

		self.init_main_model()

		if args.multiscale:
			self.init_mid_model()
			self.init_small_model()

	def embed(self, y):
		if self.dataset == 'blocks':
			obj1 = torch.cat((self.obj_embedding(y[:, 0]), torch.zeros((y.shape[0], 1), device=y.device)), dim=1)
			relation = self.relation_embedding(y[:, 1])
			obj2 = torch.cat((self.obj_embedding(y[:, 2]), torch.ones((y.shape[0], 1), device=y.device)), dim=1)
			return torch.cat((obj1, relation, obj2), dim=1)
		else:
			obj_1 = torch.cat((
				self.shape_embedding(y[:, 0]), self.size_embedding(y[:, 1]),
				self.color_embedding(y[:, 2]), self.material_embedding(y[:, 3]),
				self.index_embedding(y[:, 4])), dim=1
			)

			obj_2 = torch.cat((
				self.shape_embedding(y[:, 5]), self.size_embedding(y[:, 6]),
				self.color_embedding(y[:, 7]), self.material_embedding(y[:, 8]),
				self.index_embedding(y[:, 9])), dim=1
			)
			obj_1 = self.fc(obj_1)
			obj_2 = self.fc(obj_2)
			relation_embedding = self.relation_embedding(y[:, 10])
			return torch.cat((obj_1, obj_2, relation_embedding), dim=1)

	def init_main_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.res_3a = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_3b = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.res_4a = CondResBlock(args, classes=self.res_classes, filters=4 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_4b = CondResBlock(args, classes=self.res_classes, filters=4 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.self_attn = Self_Attn(2 * filter_dim, self.act)

		# self.fc1 = nn.Linear(filter_dim*8, 128)
		self.energy_map = nn.Linear(filter_dim * 8, 1)

	def init_mid_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.mid_conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.mid_res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.mid_res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.mid_res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.mid_res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.mid_res_3a = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.mid_res_3b = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.mid_self_attn = Self_Attn(filter_dim * 2, self.act)

		# self.mid_fc1 = nn.Linear(filter_dim*4, 128)
		self.mid_energy_map = nn.Linear(filter_dim * 4, 1)
		self.avg_pool = Downsample(channels=3)

	def init_small_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.small_conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.small_res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.small_res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.small_res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.small_res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.small_self_attn = Self_Attn(filter_dim, self.act)

		self.small_energy_map = nn.Linear(filter_dim * 2, 1)

	def init_label_map(self):
		self.map_fc1 = nn.Linear(10, 512)
		self.map_fc2 = nn.Linear(512, 512)
		self.map_fc3 = nn.Linear(512, 512)
		self.map_fc4 = nn.Linear(512, 512)

	def main_model(self, x, latent, compute_feat=False):
		x = self.act(self.conv1(x))

		x = self.res_1a(x, latent)
		x = self.res_1b(x, latent)

		x = self.res_2a(x, latent)
		x = self.res_2b(x, latent)

		if self.args.self_attn:
			x, _ = self.self_attn(x)

		x = self.res_3a(x, latent)
		x = self.res_3b(x, latent)

		x = self.res_4a(x, latent)
		x = self.res_4b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		if compute_feat:
			return x

		x = x.view(x.size(0), -1)
		# x = self.act(self.fc1(x))
		energy = self.energy_map(x)

		if self.args.square_energy:
			energy = torch.pow(energy, 2)

		if self.args.sigmoid:
			energy = F.sigmoid(energy)

		return energy

	def mid_model(self, x, latent):
		x = self.downsample(x)
		x = self.act(self.mid_conv1(x))

		x = self.mid_res_1a(x, latent)
		x = self.mid_res_1b(x, latent)

		x = self.mid_res_2a(x, latent)
		x = self.mid_res_2b(x, latent)

		if self.args.self_attn:
			x, _ = self.mid_self_attn(x)

		x = self.mid_res_3a(x, latent)
		x = self.mid_res_3b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		x = x.view(x.size(0), -1)
		# x = self.act(self.mid_fc1(x))
		energy = self.mid_energy_map(x)

		if self.args.square_energy:
			energy = torch.pow(energy, 2)

		if self.args.sigmoid:
			energy = F.sigmoid(energy)

		return energy

	def small_model(self, x, latent):
		x = self.downsample(x)
		x = self.downsample(x)

		x = self.act(self.small_conv1(x))

		x = self.small_res_1a(x, latent)
		x = self.small_res_1b(x, latent)

		if self.args.self_attn:
			x, _ = self.small_self_attn(x)

		x = self.small_res_2a(x, latent)
		x = self.small_res_2b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		x = x.view(x.size(0), -1)
		# x = self.act(self.small_fc1(x))
		energy = self.small_energy_map(x)

		if self.args.square_energy:
			energy = torch.pow(energy, 2)

		if self.args.sigmoid:
			energy = F.sigmoid(energy)

		return energy

	def label_map(self, latent):
		x = swish(self.map_fc1(latent))
		x = swish(self.map_fc2(x))
		x = swish(self.map_fc3(x))
		x = swish(self.map_fc4(x))

		return x

	def forward(self, x, latent):
		args = self.args

		if self.cond:
			latent = self.embed(latent)
		else:
			latent = None

		energy = self.main_model(x, latent)

		if args.multiscale:
			large_energy = energy
			mid_energy = self.mid_model(x, latent)
			small_energy = self.small_model(x, latent)
			energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1)

		return energy

	def compute_feat(self, x, latent):
		return self.main_model(x, None, compute_feat=True)


class ResNetCLIP(nn.Module):
	def __init__(self, args, relation=False):
		super(ResNetCLIP, self).__init__()
		self.act = swish

		self.args = args
		self.spec_norm = args.spec_norm
		self.norm = args.norm
		self.batch_size = args.batch_size
		self.dataset = args.dataset
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if self.dataset == 'visual_genome':
			if args.clip_all:
				self.res_classes = 512
			else:
				self.relation_embedding = nn.Embedding(11, 4)
				self.res_classes = 512 * 2 + 4 + 2
		elif self.dataset == 'blocks':
			if args.clip_all:
				self.res_classes = 512
			else:
				self.relation_embedding = nn.Embedding(2, 4)
				self.res_classes = 512 * 2 + 4 + 2
		else:
			if args.clip_all:
				self.res_classes = 512
			else:
				# relation embedding and object position embedding
				self.relation_embedding = nn.Embedding(7, 64)
				self.position_embedding = nn.Embedding(3, 32)
				self.res_classes = 512 * 2 + 64 + 32 * 2

		self.input_channel = 3

		self.downsample = Downsample(channels=self.input_channel)
		self.cond = args.cond

		# initialize models
		self.init_main_model()
		if args.multiscale:
			self.init_mid_model()
			self.init_small_model()

	def embed(self, y):
		# split data
		if self.dataset == 'visual_genome' or self.dataset == 'blocks':
			if self.args.clip_all:
				return y
			else:
				obj_1_embeddings, obj_2_embeddings, relations, pos_1s, pos_2s = [], [], [], [], []
				for info in y:
					obj_1, obj_2, relation, pos_1, pos_2 = info
					obj_1_embeddings.append(torch.as_tensor(obj_1))
					obj_2_embeddings.append(torch.as_tensor(obj_2))
					relations.append(relation)
					pos_1s.append(pos_1)
					pos_2s.append(pos_2)

				obj_1_embeddings = torch.cat(obj_1_embeddings, dim=0).to(self.device)
				obj_2_embeddings = torch.cat(obj_2_embeddings, dim=0).to(self.device)
				pos1_embeddings = torch.as_tensor(pos_1s, device=self.device).unsqueeze(dim=1)
				pos2_embeddings = torch.as_tensor(pos_2s, device=self.device).unsqueeze(dim=1)
				relation_embeddings = self.relation_embedding(
					torch.as_tensor(relations, dtype=torch.long, device=self.device))

				return torch.cat((
					obj_1_embeddings, pos1_embeddings, relation_embeddings,
					obj_2_embeddings, pos2_embeddings), dim=1
				)
		else:
			if self.args.clip_all:
				return y
			else:
				obj_emb_1, obj_emb_2, obj_idx_1, obj_idx_2, rel_idx = y
				# embed relations and object relative positions
				rel_embedding = self.relation_embedding(rel_idx).cuda()
				pos_1_embedding = self.position_embedding(obj_idx_1).cuda()
				pos_2_embedding = self.position_embedding(obj_idx_2).cuda()
				return torch.cat((obj_emb_1, pos_1_embedding, obj_emb_2, pos_2_embedding, rel_embedding), dim=1)

	def _embed_object(self, label):
		label = label.cuda()
		if len(label) == 1:
			# relation --> non-leaf node
			return self.relation_embedding(label)
		else:
			# object attributes --> leaf node
			obj = torch.cat((
				self.shape_embedding(label[0]), self.size_embedding(label[1]),
				self.color_embedding(label[2]), self.material_embedding(label[3])), dim=0).unsqueeze(dim=0)
			return self.fc1(obj)

	def init_main_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.res_3a = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_3b = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.res_4a = CondResBlock(args, classes=self.res_classes, filters=4 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_4b = CondResBlock(args, classes=self.res_classes, filters=4 * filter_dim, latent_dim=latent_dim,
		                           im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.self_attn = Self_Attn(2 * filter_dim, self.act)

		# self.fc1 = nn.Linear(filter_dim*8, 128)
		self.energy_map = nn.Linear(filter_dim * 8, 1)

	def init_mid_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.mid_conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.mid_res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.mid_res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.mid_res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.mid_res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.mid_res_3a = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.mid_res_3b = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
		                               im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.mid_self_attn = Self_Attn(filter_dim * 2, self.act)

		# self.mid_fc1 = nn.Linear(filter_dim*4, 128)
		self.mid_energy_map = nn.Linear(filter_dim * 4, 1)
		self.avg_pool = Downsample(channels=3)

	def init_small_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.small_conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.small_res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.small_res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.small_res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.small_res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
		                                 im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.small_self_attn = Self_Attn(filter_dim, self.act)

		self.small_energy_map = nn.Linear(filter_dim * 2, 1)

	def main_model(self, x, latent, compute_feat=False):
		x = self.act(self.conv1(x))

		x = self.res_1a(x, latent)
		x = self.res_1b(x, latent)

		x = self.res_2a(x, latent)
		x = self.res_2b(x, latent)

		if self.args.self_attn:
			x, _ = self.self_attn(x)

		x = self.res_3a(x, latent)
		x = self.res_3b(x, latent)

		x = self.res_4a(x, latent)
		x = self.res_4b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		if compute_feat:
			return x

		x = x.view(x.size(0), -1)
		energy = self.energy_map(x)

		if self.args.square_energy:
			energy = torch.pow(energy, 2)

		if self.args.sigmoid:
			energy = F.sigmoid(energy)

		return energy

	def mid_model(self, x, latent):
		x = self.downsample(x)

		x = self.act(self.mid_conv1(x))

		x = self.mid_res_1a(x, latent)
		x = self.mid_res_1b(x, latent)

		x = self.mid_res_2a(x, latent)
		x = self.mid_res_2b(x, latent)

		if self.args.self_attn:
			x, _ = self.mid_self_attn(x)

		x = self.mid_res_3a(x, latent)
		x = self.mid_res_3b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		x = x.view(x.size(0), -1)
		# x = self.act(self.mid_fc1(x))
		energy = self.mid_energy_map(x)

		if self.args.square_energy:
			energy = torch.pow(energy, 2)

		if self.args.sigmoid:
			energy = F.sigmoid(energy)

		return energy

	def small_model(self, x, latent):
		x = self.downsample(x)
		x = self.downsample(x)

		x = self.act(self.small_conv1(x))

		x = self.small_res_1a(x, latent)
		x = self.small_res_1b(x, latent)

		if self.args.self_attn:
			x, _ = self.small_self_attn(x)

		x = self.small_res_2a(x, latent)
		x = self.small_res_2b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		x = x.view(x.size(0), -1)

		energy = self.small_energy_map(x)

		if self.args.square_energy:
			energy = torch.pow(energy, 2)

		if self.args.sigmoid:
			energy = F.sigmoid(energy)

		return energy

	def forward(self, x, latent):
		args = self.args

		if self.cond:
			latent = self.embed(latent)
		else:
			latent = None

		energy = self.main_model(x, latent)

		if args.multiscale:
			large_energy = energy
			mid_energy = self.mid_model(x, latent)
			small_energy = self.small_model(x, latent)
			energy = torch.cat([small_energy, mid_energy, large_energy], dim=-1)

		return energy

	def compute_feat(self, x, latent):
		return self.main_model(x, None, compute_feat=True)


class Classifier(nn.Module):
	def __init__(self, args):
		super(Classifier, self).__init__()
		self.act = swish
		self.args = args
		self.spec_norm = args.spec_norm
		self.norm = args.norm
		self.dataset = args.dataset

		if args.dataset == 'clevr':
			self.index_embedding = nn.Embedding(3, 64)
			self.shape_embedding = nn.Embedding(4, 64)
			self.color_embedding = nn.Embedding(9, 64)
			self.size_embedding = nn.Embedding(3, 64)
			self.material_embedding = nn.Embedding(3, 64)
			self.relation_embedding = nn.Embedding(7, 512)
			self.fc = nn.Linear(320, 512)
			self.res_classes = 1536
			self.input_channel = 3
		elif args.dataset == 'igibson':
			self.index_embedding = nn.Embedding(3, 64)
			self.shape_embedding = nn.Embedding(5, 64)
			self.color_embedding = nn.Embedding(6, 64)
			self.size_embedding = nn.Embedding(2, 64)
			self.material_embedding = nn.Embedding(4, 64)
			self.relation_embedding = nn.Embedding(7, 512)
			self.fc = nn.Linear(320, 512)
			self.res_classes = 1536
			self.input_channel = 3
		elif args.dataset == 'blocks':
			self.obj_embedding = nn.Embedding(4, 16)  # 31 objects
			self.relation_embedding = nn.Embedding(2, 16)  # 11 relations
			# 3 * 16 + 2 indices embedding
			self.res_classes = 50
			self.input_channel = 3
		else:
			raise NotImplementedError

		self.downsample = Downsample(channels=self.input_channel)
		self.init_main_model()

	def embed(self, y):
		if self.dataset == 'blocks':
			obj1 = torch.cat((self.obj_embedding(y[:, 0]), torch.zeros((y.shape[0], 1), device=y.device)), dim=1)
			relation = self.relation_embedding(y[:, 1])
			obj2 = torch.cat((self.obj_embedding(y[:, 2]), torch.ones((y.shape[0], 1), device=y.device)), dim=1)
			return torch.cat((obj1, relation, obj2), dim=1)
		else:
			obj_1 = torch.cat((self.shape_embedding(y[:, 0]), self.size_embedding(y[:, 1]),
							   self.color_embedding(y[:, 2]), self.material_embedding(y[:, 3]),
							   self.index_embedding(y[:, 4])), dim=1)

			obj_2 = torch.cat((self.shape_embedding(y[:, 5]), self.size_embedding(y[:, 6]),
							   self.color_embedding(y[:, 7]), self.material_embedding(y[:, 8]),
							   self.index_embedding(y[:, 9])), dim=1)
			obj_1 = self.fc(obj_1)
			obj_2 = self.fc(obj_2)
			relation_embedding = self.relation_embedding(y[:, 10])

			return torch.cat((obj_1, obj_2, relation_embedding), dim=1)

	def init_main_model(self):
		args = self.args
		filter_dim = args.filter_dim
		latent_dim = args.filter_dim
		im_size = args.im_size

		self.conv1 = nn.Conv2d(self.input_channel, filter_dim, kernel_size=3, stride=1, padding=1)
		self.res_1a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
								   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_1b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
								   im_size=im_size, rescale=False, spec_norm=self.spec_norm, norm=self.norm)

		self.res_2a = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
								   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_2b = CondResBlock(args, classes=self.res_classes, filters=filter_dim, latent_dim=latent_dim,
								   im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.res_3a = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
								   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_3b = CondResBlock(args, classes=self.res_classes, filters=2 * filter_dim, latent_dim=latent_dim,
								   im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.res_4a = CondResBlock(args, classes=self.res_classes, filters=4 * filter_dim, latent_dim=latent_dim,
								   im_size=im_size, downsample=False, spec_norm=self.spec_norm, norm=self.norm)
		self.res_4b = CondResBlock(args, classes=self.res_classes, filters=4 * filter_dim, latent_dim=latent_dim,
								   im_size=im_size, rescale=True, spec_norm=self.spec_norm, norm=self.norm)

		self.self_attn = Self_Attn(2 * filter_dim, self.act)

		self.energy_map = nn.Linear(filter_dim * 8, 1)

	def main_model(self, x, latent):
		x = self.act(self.conv1(x))

		x = self.res_1a(x, latent)
		x = self.res_1b(x, latent)

		x = self.res_2a(x, latent)
		x = self.res_2b(x, latent)

		x, _ = self.self_attn(x)

		x = self.res_3a(x, latent)
		x = self.res_3b(x, latent)

		x = self.res_4a(x, latent)
		x = self.res_4b(x, latent)
		x = self.act(x)

		x = x.mean(dim=2).mean(dim=2)

		x = x.view(x.size(0), -1)
		energy = self.energy_map(x)

		return torch.sigmoid(energy)

	def forward(self, x, y):
		energy = self.main_model(x, self.embed(y))
		return energy