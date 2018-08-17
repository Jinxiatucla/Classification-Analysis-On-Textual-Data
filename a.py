import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_20newsgroups

def fetch_data():
	categories = np.array(['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'])
	graphics_train = fetch_20newsgroups(subset = 'train', categories = categories, shuffle = True, random_state = 42)
	return (categories, graphics_train)

def get_frequency(categories, graphics_train):
	category_number = np.zeros(len(categories), dtype = np.int16)
	length = len(graphics_train.data)	

	for i in range(0, length):
		cate = graphics_train.target_names[graphics_train.target[i]]
		if cate in categories:
			index = np.argwhere(categories == cate)
			category_number[index] += 1
	for x in category_number:
		print x

	return category_number 

def plot(categories, category_number):
	ind = np.linspace(0.5, 9.5, len(categories))
	fig = plt.figure(1)
	ax = fig.add_subplot(111)
	ax.bar(ind - 0.2, category_number, 0.4)
	ax.set_xticks(ind)
	ax.set_xticklabels(categories)
	plt.grid(True)
	plt.show()
	plt.close()

if __name__ == "__main__":
	categories, graphics_train = fetch_data()
	category_number = get_frequency(categories, graphics_train)
	plot(categories, category_number)	


