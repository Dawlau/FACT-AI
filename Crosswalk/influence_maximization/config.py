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
			self.filename = os.path.join(self.filename, "synthetic_n500_Pred0.7_Phom0.025_Phet0.001")
		elif self.dataset == "synth3":
			self.weight = 0.03
			self.filename = os.path.join(self.filename, "synthetic_3g_n500_Pred0.6_Pblue0.25_Prr0.025_Pbb0.025_Pgg0.025_Prb0.001_Prg0.0005_Pbg0.0005")
		elif self.dataset == "twitter":
			self.weight = 0.03
			self.filename = os.path.join(self.filename, self.dataset)
		else:
			raise Exception("Invalid dataset provided")