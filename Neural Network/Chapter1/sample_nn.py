import numpy
import scipy.special

class nn:
	#define the nunber of input nodes, hidden nodes and output nodes and learning rate.
	def __init__(self, inputnodes, hiddennodes, outputnodes, lr):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = lr
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
		#define the sigmoid function as the activation function
		self.activation_function = lambda x: scipy.special.expit(x)

	def train(self, input, target):
		inputs = numpy.array(input, ndim=2).T
		targets = numpy.array(target, ndim=2).T
		#calculate the hidden input by multipling the weight
		hidden_input = numpy.dot(self.wih, input)
		#calculate the output after the hidden layer by using the activation function
		hidden_output = self.activation_function(hidden_input)

		#calculate the output layer's input
		out_input = numpy.dot(self.who, hidden_output)
		#obtain the final output after the whole network
		output = self.activation_function(out_input)

		#calculate the error of output
		error = targets - output
		#calculate the error of hidden layer according to the weight ratio
		hidden_error = numpy.dot(self.who.T, error)
		#update the weights
		self.who += self.lr*numpy.dot((error*output*(1.0 - output)), numpy.transpose(hidden_output))
		self.wih += self.lr*numpy.dot((error*hidden_output*(1.0 - hidden_output)), numpy.transpose(inputs))

	def query(self, input):
		#get the output by enter the input, whicn is used as test the network model
		inputs = numpy.array(input, ndim=2).T

		hidden_input = numpy.dot(self.wih, input)
		hidden_output = self.activation_function(hidden_input)

		out_input = numpy.dot(self.who, hidden_output)
		output = self.activation_function(out_input)




