from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib

def build_dataset(num):
	x = np.random.uniform(-8, 8, num)
	K = Kernel_mat(x, 1.5, 2)
	y = np.random.multivariate_normal(np.zeros(num), K)
	e = np.random.normal(0, 0.1, num)

	return x, y + e

def kappa(x, y, scale, sensitivity):
	'''
	The kernel
	args: 
		x, y are scalars
	'''
	return np.exp(-np.power(np.abs(x - y), sensitivity) / scale)

def Kernel_mat(x, scale, sensitivity):
	'''
	args:
		x is a vector
	'''
	K = np.zeros((len(x), len(x)))
	for i in range(len(x)):
		for j in range(i + 1):
			if i == j:
				K[i, j] = kappa(x[i], x[j], scale, sensitivity) / 2
			else:
				K[i, j] = kappa(x[i], x[j], scale, sensitivity)
	
	K = K + K.T 
	return K 

def GP_regression(x_test, x_train, y_train, sigma_y = 0.1, scale = 1, sensitivity = 2):
	K = Kernel_mat(x, scale, sensitivity)
	K_y = K + np.power(sigma_y, 2) * np.identity(len(x_train))
	L = np.linalg.cholesky(K_y)
	alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

	mu_n = np.zeros(len(x_test))
	sigma_n = np.zeros(len(x_test))

	for i in range(len(mu_n)):
		sim = np.array([kappa(x_test[i], j, scale, sensitivity) for j in x_train])
		mu_n[i] = sim.dot(alpha)
		beta = np.linalg.solve(L, sim)
		sigma_n[i] = np.sqrt(kappa(x_test[i], x_test[i], scale, sensitivity) - beta.dot(beta))

	return mu_n, sigma_n

def K_y_grad_scale(x, sigma_y, scale, sensitivity):
	'''
	Args: x is a vector of one dimensional observations
	return: the gradient of K_y with respect to scale, it is a matrix
	'''
	if len(x) == 1:
		raise Exception('The length of x should be greater than 1')

	dist_mat = np.zeros((len(x), len(x)))
	for i in range(1, len(x)):
		for j in range(i):
			dist_mat[i,j] = np.power(np.abs(x[i] - x[j]), sensitivity)
	dist_mat = dist_mat + dist_mat.T

	grad_mat = np.exp(- dist_mat / scale) * dist_mat / scale / scale
	return grad_mat

def get_grad(x, y, sigma_y, scale, sensitivity):
	'''
	get the gradient of scale
	'''

	K_y = Kernel_mat(x, scale, sensitivity) + np.power(sigma_y, sensitivity) * np.identity(len(x))
	L = np.linalg.cholesky(K_y)
	alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
	grad_mat = K_y_grad_scale(x, sigma_y, scale, sensitivity)
	grad_1 = alpha.dot(grad_mat).dot(alpha)
	grad_2 = np.trace(np.linalg.solve(L.T, np.linalg.solve(L, grad_mat)))
	return grad_1 - grad_2

def optimize_scale(x, y, sigma_y, sensitivity, learning_rate, epoch):
	scale = np.random.uniform()
	scale_vec = [scale]
	for i in range(epoch):
		grad = get_grad(x, y, sigma_y, scale, sensitivity)
		if grad < 1e-10:
			break
		scale += learning_rate * grad 
		scale_vec.append(scale)
	if i == 999:
		print 'The result may not be accurate'
	# plt.plot(scale_vec)
	# plt.show()
	return scale


x, y = build_dataset(20)

##------------------ observe dataset ------------------------------
# plt.scatter(x, y)
# plt.show()

##------------------- observe kernel matrix -----------------------
# K = Kernel_mat(x, scale, sensitivity)
# plt.pcolor(K, cmap = matplotlib.cm.Blues)
# plt.show()

##------------------ observe regression ---------------------------
# plt.scatter(x, y)
# base_x = np.arange(-3, 3, 0.01)
# base_y = np.sin(base_x)
# mu_n, sigma_n = GP_regression(base_x, x, y)
# plt.plot(base_x, base_y, color = 'y', linestyle = '--', label = 'True')
# plt.plot(base_x, mu_n, color = 'r', linestyle = '-', label = 'Prediction')
# plt.plot(base_x, mu_n + 2 * sigma_n, color = 'b', linestyle = ':')
# plt.plot(base_x, mu_n - 2 * sigma_n, color = 'b', linestyle = ':')
# plt.legend()
# plt.show()

##------------------ observe parameter optimization ---------------
optimal = np.zeros(100)
for i in range(100):
	optimal[i] = optimize_scale(x, y, 0.1, 2, 0.1, 1000)
	print 'iter %d' %i
print np.median(optimal), np.std(optimal)
plt.hist(optimal)
plt.show()