from collections import Counter 
from scipy.stats import geom
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit



def read_csv_file(file_name):
	"""Function meant to open, read, and close a csv file.
	Returns the contents of the file as a list of lists.

	Args:
		file_name ([str]): [an image file]

	Returns:
		[list]: [a list of lists, where each inner list contains the contents of a row]
	"""	
	file = open(file_name, newline = '')

	data = [line.rstrip("\n").split(",") for line in file]

	file.close()

	return data


def first_digit_distribution(data_results):
	"""Creates a list of the first digits found in a list of numbers, numbers as strings

	Args:
		data_results ([list]): [a list of strings]

	Returns:
		[list]: [a list of the first digit in each string number]
	"""
	data_first_digit = []
	for i in data_results:
		num_strings = [int(j) for j in i if j]
		for j in num_strings:
			first_digit = 0
			while first_digit % 10 != j % 10:
				first_digit += 1
			data_first_digit.append(first_digit)
	return data_first_digit


def frequency_distribution(data_results):
	"""Returns the frequency distribution for each data point in the results

	Args:
		data_results ([list]): [data to be analyzed]

	Returns:
		[int]: [data frequency]
	"""
	data_first_digit = first_digit_distribution(data_results)
	freq_dist = Counter(data_first_digit)
	freq_dist = {i:freq_dist[1:][i]/len(data_first_digit)*100 for i in freq_dist[1:]}

	return freq_dist

def benfords_law(n):
	"""Calculates the number distrubiton for digits 1-9 based on the distribution's congruency with the theory of Benford's law

	Args:
		n ([int]): [the coordinate of the x axis used to compute the y axis of the distribution]

	Returns:
		[int]: [the y coordinate of the distribution]
	"""
	return np.log10(1 + (1/n))

def compare_distributions(benford_data,frequency_data):
	"""Returns monte carlo simulated distributions of data following both a geometric and a benford curve by choosing data points at random, and checks goodness of fit by checking chi squared.

	Args:
		benford_data ([dict]): [key is digit (1-9), the value is the percentage rounded to 2 digits]
		frequency_data([dict]): [key is digit (1-9), the value is the percentage rounded to 2 digits]

	Returns:
		[bool]: [whether or not the distributions are similar]
	"""
	first_digit_hits = []
	for i in range(10000):
		first_digit_hits.append(min(frequency_data.keys(), key = lambda x: abs(frequency_data[x] - 88)))	#calculates abs value of probabilities in dict, then chooses min key, corresponding to a reference digit (1-9)

	benford_distribution = []
	for i in range(1,10):
		benford_distribution.extend([i]*int(first_digit_hits.count(i)*benford_data[i]))
	chi_squared = 0
	for i in range(1,10):
		chi_squared += ((benford_distribution.count(i) - 100 * (benford_data[i]))**2) / (100 * (benford_data[i]))
	if chi_squared < 21:
		return "Data is congruent with Benfords law, your data follows closely similar trends"
	else:
		return "Data is inconclusive, your data is not very congruent"

def poisson_distribution(freq_dist):
	"""Using MLE to calculate the poisson distribution for all probabilties in the freq_dist dict

	Args:
		freq_dist ([dict]): [key is the powerball (1-69), value is the probabilities in the strength of 2 digits(0.24%)]

	Returns:
		[list of dict]: [A list of dictionaries with keys of integers 1-69, values of the probability of that key drawn]
	"""
	max_probabilities = {}
	poissondist = {}
	for i in freq_dist:
		for j in range(1,11):
			poissondist[j] = (freq_dist[i]**j*np.exp(-freq_dist[i]))/np.math.factorial(j)
		if len(max_probabilities) > 5:
			del max_probabilities[min(max_probabilities,key=max_probabilities.get)],
		max_probabilities[max(poissondist,key=poissondist.get)] = max(poissondist.values())
		poissondist = {}
	return  max_probabilities



def plot_data(x_data,y_data,file_name, x_label, y_label,title, log = 0, color = 'ro',ma = None):
	'''Plots a graph based on the data passed in 

	Args:
		x_data ([list]): [The x data to be used]
		y_data ([list]): [The y data to be used]
		file_name ([str]): [The file name to be passed]
		x_label ([str]): [title to be used for x-axis]
		y_label ([str]): [title to be used for y-axis]
		title ([str]): [title to be used for the graph]
		[log (0 or 1)]: [how many log functions]

	Returns:
		[png]: [A plot to be printed on a png file]

	'''


	plt.clf()
	fig1 = plt.figure()
	xdata = x_data
	ydata = y_data
	plt.xlabel(x_label)
	plt.xlim(0.5, 6.5)
	plt.ylabel(y_label)
	plt.title(title)
	for i in range(log):
		plt.xscale('log')
	if ma is None:
		plt.scatter(xdata, ydata, s=10, c='b', marker="s", label='skitscat', alpha=0.5)
	else:
		bins = [i for i in range(9,69)]	#bins of data
		y = geom.pmf(bins,1/23)
		plt.bar(bins, y)
		plt.plot(,[np.mean(freq.values().values)/2 for _ in range(69)])
		plt.scatter(xaxis,list(freq[1:]))
	plt.savefig(file_name)
	return plt.show()
	#plt.plot(xdata, ydata, color)


if __name__ == '__main__':
	#xdata = list(range(1,10))
	#bb_data = [int(10*i[1])*benfords_law(i[0]) for i in enumerate(xdata,start = 1)]
	#fig1 = plot_data(x_data = xdata, y_data = bb_data, file_name = 'Benfords_Law.png', x_label = '1st Digits', y_label = 'Relative Frequency (%) of Digits', title = "Relative Frequencies of First Digits in Powerball Lottery Results 2019-2020")
	#powerball_file = read_csv_file('powerballs.csv')
	powerball_file = read_csv_file('winningtitanic.csv')
	#for i in powerball_file:
	# 	i[1], i[2], i[4] = int(i[1]), int(i[2]), int(i[4])
   
	powerball_dict = [dict(enumerate(powerball_file[i], start = 1)) for i in range(890)]
	powerball_dictionary = {i[dict(enumerate(powerball_file[i], start = 1))]: i[dict(enumerate(powerball_file[i], start = 1))] for i in powerball_dict}

    #powerball_dict = {i[0]: i[1:] for i in powerball_file}
	for i in powerball_dict:
		i[1], i[2], i[4] = int(i[1]), int(i[2]), int(i[4])
   
	freq = frequency_distribution([i[1:] for i in powerball_dict[1:900]])

	fatmer = max(freq, key = freq.get)
	ways = np.math.factorial(59)/(np.math.factorial(5)*np.math.factorial(54))
	w = ways*freq[1:]
	print(freq)
	print(int(sum(w)))
	
	print(w)
