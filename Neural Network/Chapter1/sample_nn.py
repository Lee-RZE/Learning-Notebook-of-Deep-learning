import numpy
import scipy.special
class nn:
	def __init__(self, inputnodes, hiddennodes, outputnodes, lr):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = lr
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		self.activation_function = lambda x: scipy.special.expit(x)

	def train(self, input, target):
		inputs = numpy.array(input, ndim=2).T
		targets = numpy.array(target, ndim=2).T
		hidden_input = numpy.dot(self.wih, input)
		hidden_output = self.activation_function(hidden_input)

		out_input = numpy.dot(self.who, hidden_output)
		output = self.activation_function(out_input)

		error = targets - output
		hidden_error = numpy.dot(self.who.T, error)
		self.who += self.lr*numpy.dot((error*output*(1.0 - output)), numpy.transpose(hidden_output))
		self.wih += self.lr*numpy.dot((error*hidden_output*(1.0 - hidden_output)), numpy.transpose(inputs))

	def query(self, input):
		inputs = numpy.array(input, ndim=2).T

		hidden_input = numpy.dot(self.wih, input)
		hidden_output = self.activation_function(hidden_input)

		out_input = numpy.dot(self.who, hidden_output)
		output = self.activation_function(out_input)




