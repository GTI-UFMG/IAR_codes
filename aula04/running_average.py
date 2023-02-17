##########################################
class RunningAverage(object):
	def __init__(self, N):
		self.N = N
		self.vals = []
		self.num_filled = 0

	def push(self, val):
		if self.num_filled == self.N:
			self.vals.pop(0)
			self.vals.append(val)
		else:
			self.vals.append(val)
			self.num_filled += 1

	def get(self):
		return float(sum(self.vals)) / self.num_filled
