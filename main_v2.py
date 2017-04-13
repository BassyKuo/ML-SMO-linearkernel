from copy import deepcopy
import sys
import argparse
import numpy as np

class SMO:
	def __init__ (self, data, label, w, b, C):
		self.data	= data	# x (MxN)
		self.label	= label	# y (Mx1)
		self.w		= w		# weight (1xN)
		self.b		= b		# bias
		self.C		= C		# upper bound parameter
		self.lam	= np.zeros(len(label))	# Lagrange multipliers lambda (N,)
		self.k		= data.dot(data.T)

	def change_dataset (self, data, label):
		self.data	= data
		self.label	= label

	def Kernel (self, Xi, Xj):
		"""
							[ Kernel(x1, Xj) ]	 [ Kernel(x1, x1) Kernel(x1, x2) .... Kernel(x1, xm) ]
			Kernel(Xi,Xj) = [	.... ....	 ] = [   .... ....	    .... ....	 ....   .... ....	 ]
							[ Kernel(xm, Xj) ]	 [ Kernel(xm, x1) Kernel(xm, x2) .... Kernel(xm, xm) ]

			Kernel(xk,Xj) = [ Kernel(xk, x1) .... Kernel(xk, xm) ]

							[ Kernel(x1, xk) ]
			Kernel(Xi,xk) = [	.... ....	 ]
							[ Kernel(xm, xk) ]

		- Xi, Xj: data[1:m,:]	or
		- xk	: data[k,:]
		=Return	: a (m,m) matrix or (m, ) matrix
		"""
		return Xi.dot(Xj.T)

	def H_x (self, i):
		"""
		H_(xi) = reduce_sum(lam * y * Kernel(xi,x)) + b
		"""
		# return (self.lam * self.label.T).dot(self.Kernel(self.data[i], self.data)) + self.b
		return (self.lam * self.label.T).dot(self.k[i,:]) + self.b

	def takeStep (self, i, j):
		if (i == j):
			return False
		eps	 = 0.001
		lam1 = self.lam[i]
		lam2 = self.lam[j]
		y1	 = self.label[i,0]
		y2	 = self.label[j,0]
		E1	 = self.H_x(i) - y1
		E2	 = self.H_x(j) - y2
		s	 = y1 * y2
		L	 = max(0, lam2 - lam1) if s < 0 else max(0, lam2 + lam1 - self.C)
		H	 = min(self.C, self.C + lam2 - lam1) if s < 0 else min(self.C, lam2 + lam1)
		if L == H:
			return False
		K11	 = self.k[i,i]
		K12	 = self.k[i,j]
		K22	 = self.k[j,j]
		eta	 = K11 + K22 - 2 * K12
		if (eta > 0):
			lam2_new = np.clip(lam2 + y2 * (E1 - E2) / eta, L, H)
		else:
			L_psi = y2 * (E1 - E2) * L
			H_psi = y2 * (E1 - E2) * H
			if (L_psi < H_psi - eps):
				lam2_new = L
			elif (L_psi > H_psi + eps):
				lam2_new = H
			else:
				return False
		if (abs(lam2 - lam2_new) < eps * (lam2 + lam2_new + eps)):
			return False
		lam1_new = lam1 + s * (lam2 - lam2_new) 
		########################################################################
		## Update b
		########################################################################
		b1 =  -(E1 + y1 * (lam1_new - lam1) * K11 + y2 * (lam2_new - lam2) * K12) + self.b
		b2 =  -(E2 + y1 * (lam1_new - lam1) * K12 + y2 * (lam2_new - lam2) * K22) + self.b
		if (0 < lam1_new and lam1_new < self.C):
			self.b = b1
		elif (0 < lam2_new and lam2_new < self.C):
			self.b = b2
		else:
			self.b = (b1 + b2) / 2
		# print("Update b = ", self.b)
		########################################################################
		## Update w
		########################################################################
		self.w = self.w + y1 * (lam1_new - lam1) * self.data[i] + y2 * (lam2_new - lam2) * self.data[j]
		# print("Update w = ", self.w)
		########################################################################
		## Update error cache without bias F
		########################################################################
		# self.F = self.F + y1 * (lam1_new - lam1) * self.k[i,:] \
						# + y2 * (lam2_new - lam2) * self.k[j,:]
		########################################################################
		## Update lambda_i and lambda_j
		########################################################################
		self.lam[i] = lam1_new
		self.lam[j] = lam2_new
		# print("Update lam[%s] = %s		(old: %s)" % (i, lam1_new, lam1))
		# print("Update lam[%s] = %s		(old: %s)" % (j, lam2_new, lam2))
		return True

	def examine (self, j, tol=0.001):
		# print("choose j = ", j)
		y2	 = self.label[j,0]
		lam2 = self.lam[j]
		E2	 = self.H_x(j) - y2
		# print("E(%s) : %s" % (j, E2))
		r2	 = E2 * y2
		if ((r2 < -tol and lam2 < self.C) or (r2 > tol and lam2 > 0)):
			######
			## cached error is too big, must to update
			#####
			out_of_bound = (self.lam < 0) + (self.lam > self.C)
			if (np.sum(out_of_bound) > 1):
				E = (self.lam * self.label.T).dot(self.k[:,:]) + self.b - self.label.reshape(-1)
				i = np.argmax(E) if E2 <= 0 else np.argmin(E)
				if (self.takeStep(i,j)):
					# print("///////  CASE1 : choose i = ", i)
					# print("-----------UPDATE-----------")
					return 1
			out_index = np.arange(len(self.lam))[out_of_bound]
			np.random.shuffle(out_index)
			for i in out_index:
				if (self.takeStep(i,j)):
					# print("///////  CASE2 : choose i = ", i)
					# print("-----------UPDATE-----------")
					return 1
			all_index = np.arange(len(self.lam))
			np.random.shuffle(all_index)
			for i in all_index:
				if (self.takeStep(i,j)):
					# print("///////  CASE3 : choose i = ", i)
					# print("-----------UPDATE-----------")
					return 1
		return 0


class Hypothesis:
	def __init__ (self, w, b):
		self.w = w
		self.b = b
	def sgn (self, data):
		guess = data[:].dot(self.w.T) + self.b
		pos = (1)  * (guess > 0)
		neg = (-1) * (guess < 0)
		self.label = pos + neg
		return self.label
	def error (self, c_label, data):
		h_label = self.sgn(data)
		return np.mean(c_label != h_label)
	def update (self, w, b):
		self.w = w
		self.b = b

def load_data (train_csv, test_csv):
	"""
	Usage:
		train_x, train_y, test_x, test_y = load_data("train.csv", "test.csv")

	-train	: train data
	-test	: test data
	"""
	train = np.genfromtxt(train_csv, delimiter=',')
	test  = np.genfromtxt(test_csv, delimiter=',')

	y_tr = np.array(train[:,:1])
	x_tr = np.array(train[:,1:])
	y_te = np.array(test[:,:1])
	x_te = np.array(test[:,1:])
	return x_tr, y_tr, x_te, y_te

def shuffle_union (x, y):
	index = np.random.permutation(len(x))
	return x[index], y[index]

def in_range(lower, upper):
	class RequiredRange(argparse.Action):
		def __call__ (self, parser, args, values, option_str=None):
			if lower > values or values > upper:
				msg='argument {f} requires between {lower} and {upper}'.format(f=self.dest, lower=lower, upper=upper)
				raise argparse.ArgumentTypeError(msg)
			setattr(args, self.dest, values)
	return RequiredRange

def ProgParser ():
	"""
	Command line:
		python3 <this_file> --trainset {train.csv} --testset {test.csv} --c {free_parameter_c} --n {number_of_fold} --tol {tolerance}
	"""
	program_str = sys.argv[0]
	parser = argparse.ArgumentParser(prog=program_str)
	parser.add_argument('--version',  action='version', version='%(prog)s 2.0')
	parser.add_argument('--trainset', default='messidor_features_training.csv', help='a training file (.csv)')
	parser.add_argument('--testset',  default='messidor_features_testing.csv',  help='a testing file (.csv)')
	parser.add_argument('--c',		  default=0.7,   help='a free parameter C (between 0 and 1) [default: %(default)s]', type=float, action=in_range(0,1))
	parser.add_argument('--n',		  default=5,     help='the number of fold (5 or 10) [default: %(default)s]', type=int, choices=[5,10])
	parser.add_argument('--tol',	  default=0.001, help='the tolerance of support vector between support hyperplane [default: %(default)s]', type=float)
	args = parser.parse_args()
	# print(args)
	return args

################################################################################################################################################################
if __name__ == '__main__':
	args = ProgParser()

	C = args.c
	n = args.n
	tol = args.tol
	f = open('%s-c%s_n%s_tol%s.txt' % (sys.argv[0], C, n, tol), 'w')
	xtr, ytr, xte, yte = load_data(args.trainset, args.testset)

	## Remap: Y = {-1, +1}
	ytr = (ytr * 2 - (max(ytr) + min(ytr))) / (max(ytr) - min(ytr))
	yte = (yte * 2 - (max(yte) + min(yte))) / (max(yte) - min(yte))

	xtr, ytr = shuffle_union(xtr, ytr)
	xte, yte = shuffle_union(xte, yte)
	fold_size = len(xtr) // n
	R_i = 0
	R_opt = 1

	## Initialization of weight, bias, lambda
	w	= np.zeros((1,xtr.shape[1]))	# weight: (1xN)
	#lam = np.zeros(len(ytr) - fold_size)		# Lagrange multipliers lambda: (m, ) 
	b	= 0								# bias: (constant)
	h	= Hypothesis(w,b)

	for i in range(n):
		###################################
		### Cross-validation training
		###################################
		xcv = np.concatenate((xtr[:i * fold_size], xtr[(i+1) * fold_size:]), axis=0)
		ycv = np.concatenate((ytr[:i * fold_size], ytr[(i+1) * fold_size:]), axis=0)
		trainer = SMO(xcv, ycv, w, b, C)

		numChanged = 0
		toCheckAll = True
		epoch = 0
		threshold = len(ycv) * 0.00005 / tol
		while ((numChanged > threshold or toCheckAll) and epoch < 10000):
			epoch += 1
			numChanged = 0
			if (toCheckAll):
				for j in range(len(ycv)):
					numChanged += trainer.examine(j,tol)
					# print("[epoch %s] numChanged: %s" % (epoch, numChanged))
			else:
				out_of_bound = (trainer.lam > 0) + (trainer.lam < C)
				out_index = np.arange(len(trainer.lam))[out_of_bound]
				for j in out_index:
					numChanged += trainer.examine(j,tol)
					# print("[epoch %s] numChanged: %s" % (epoch, numChanged))
			w = trainer.w
			b = trainer.b
			h.update(w,b)
			print("[epoch %s] numChanged: %s" % (epoch, numChanged))
			print("[epoch %s] Error: %s" % (epoch, h.error(ycv, xcv)))
			f.write("[epoch %s] numChanged: %s\n" % (epoch, numChanged))
			f.write("[epoch %s] Error: %s\n" % (epoch, h.error(ycv, xcv)))
			if (toCheckAll):
				toCheckAll = False
			elif (numChanged <= threshold):
				toCheckAll = True

		else:
			print("Training stop!")
			f.write("Training stop!\n")
			h.update(trainer.w, trainer.b)
		
		###################################
		### Cross-validation testing
		###################################
		xcv = xtr[i * fold_size:(i+1) * fold_size]
		ycv = ytr[i * fold_size:(i+1) * fold_size]
		print("=============================================")
		print("[%sth] CV-Test Result:" % i)
		print("+ w: ", h.w)
		print("+ b: ", h.b)
		print("+ Testing error: ", h.error(ycv,xcv))
		print("=============================================")
		f.write("=============================================\n")
		f.write("[%sth] CV-Test Result:\n" % i)
		f.write("+ w: %s\n" % h.w)
		f.write("+ b: %s\n" % h.b)
		f.write("+ Testing error: %s\n" % h.error(ycv,xcv))
		f.write("=============================================\n")
		R_i += h.error(ycv, xcv)
	###///// End of for(i in fold) /////###

	R_cv = R_i / n
	if R_cv < R_opt:
		C_opt = C
		h_opt = deepcopy(h)

	print("=============================================")
	print("Final Result:")
	print("+ fold: ", n)
	print("+ Optimal C: ", C_opt)
	print("+ optimal w: ", h_opt.w)
	print("+ optimal b: ", h_opt.b)
	print("+ Testing error: ", h.error(yte,xte))
	print("=============================================")
	f.write("=============================================\n")
	f.write("Final Result:\n")
	f.write("+ fold: %s\n" % n)
	f.write("+ Optimal C: %s\n" % C_opt)
	f.write("+ optimal w: %s\n" % h_opt.w)
	f.write("+ optimal b: %s\n" % h_opt.b)
	f.write("+ Testing error: %s\n" % h.error(yte,xte))
	f.write("=============================================\n")
	f.close()
