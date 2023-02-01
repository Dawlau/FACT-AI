import numpy as np
import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class infMaxConfig(object):
	def __init__(self, args):
		print(args)
		self.dataset = args.dataset
		self.filename = os.path.join(ROOT_DIR, "data", self.dataset)

		if self.dataset == "rice_subset":
			self.weight = 0.01
			self.filename = os.path.join(self.filename, self.dataset)
		elif self.dataset == "synth2":
			self.weight = 0.03
			self.filename = os.path.join(self.filename, self.dataset)
		elif self.dataset == "synth3":
			self.weight = 0.03
			self.filename = os.path.join(self.filename, self.dataset)
		elif self.dataset == "twitter":
			self.weight = 0.03
			self.filename = os.path.join(self.filename, self.dataset)
		else:
			raise Exception("Invalid dataset provided")