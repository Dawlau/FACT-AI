import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class infMaxConfig(object):
	def __init__(self, args):
		self.data_path = args.data_path
		self.dataset = args.dataset
		self.filename = os.path.join(ROOT_DIR, self.data_path, self.dataset)

		if self.dataset == "rice_subset":
			self.weight = 0.01
			self.filename = os.path.join(self.filename, self.dataset)
		else:
			self.weight = 0.03
			self.filename = os.path.join(self.filename, self.dataset)
