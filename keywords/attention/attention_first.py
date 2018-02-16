from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed
from attention_decoder import AttentionDecoder
 
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import seq2seq
from seq2seq.models import AttentionSeq2Seq


def dot_product(x, kernel):
	"""
	Wrapper for dot product operation, in order to be compatible with both
	Theano and Tensorflow
	Args:
		x (): input
		kernel (): weights
	Returns:
	"""
	if K.backend() == 'tensorflow':
		return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
	else:
		return K.dot(x, kernel)

class Attention(Layer):
	def __init__(self,
				 W_regularizer=None, b_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 bias=True,
				 return_attention=False,
				 **kwargs):
		"""
		Keras Layer that implements an Attention mechanism for temporal data.
		Supports Masking.
		Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
		# Input shape
			3D tensor with shape: `(samples, steps, features)`.
		# Output shape
			2D tensor with shape: `(samples, features)`.
		:param kwargs:
		Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
		The dimensions are inferred based on the output shape of the RNN.
		Note: The layer has been tested with Keras 1.x
		Example:
		
			# 1
			model.add(LSTM(64, return_sequences=True))
			model.add(Attention())
			# next add a Dense layer (for classification/regression) or whatever...
			# 2 - Get the attention scores
			hidden = LSTM(64, return_sequences=True)(words)
			sentence, word_scores = Attention(return_attention=True)(hidden)
		"""
		self.supports_masking = True
		self.return_attention = return_attention
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1],),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)
		if self.bias:
			self.b = self.add_weight((input_shape[1],),
									 initializer='zero',
									 name='{}_b'.format(self.name),
									 regularizer=self.b_regularizer,
									 constraint=self.b_constraint)
		else:
			self.b = None

		self.built = True

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		eij = dot_product(x, self.W)

		if self.bias:
			eij += self.b

		eij = K.tanh(eij)

		a = K.exp(eij)

		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			a *= K.cast(mask, K.floatx())

		# in some cases especially in the early stages of training the sum may be almost zero
		# and this results in NaN's. A workaround is to add a very small positive number eps to the sum.
		# a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		weighted_input = x * K.expand_dims(a)

		result = K.sum(weighted_input, axis=1)

		if self.return_attention:
			return [result, a]
		return result

	def compute_output_shape(self, input_shape):
		if self.return_attention:
			return [(input_shape[0], input_shape[-1]),
					(input_shape[0], input_shape[1])]
		else:
			return input_shape[0], input_shape[-1]

class AttentionWithContext(Layer):
	"""
	Attention operation, with a context/query vector, for temporal data.
	Supports Masking.
	Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
	"Hierarchical Attention Networks for Document Classification"
	by using a context vector to assist the attention
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
	# Output shape
		2D tensor with shape: `(samples, features)`.
	How to use:
	Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
	The dimensions are inferred based on the output shape of the RNN.
	Note: The layer has been tested with Keras 2.0.6
	Example:
		model.add(LSTM(64, return_sequences=True))
		model.add(AttentionWithContext())
		# next add a Dense layer (for classification/regression) or whatever...
	"""

	def __init__(self,
				 W_regularizer=None, u_regularizer=None, b_regularizer=None,
				 W_constraint=None, u_constraint=None, b_constraint=None,
				 bias=True, **kwargs):

		self.supports_masking = True
		self.init = initializers.get('glorot_uniform')

		self.W_regularizer = regularizers.get(W_regularizer)
		self.u_regularizer = regularizers.get(u_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.u_constraint = constraints.get(u_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		super(AttentionWithContext, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3

		self.W = self.add_weight((input_shape[-1], input_shape[-1],),
								 initializer=self.init,
								 name='{}_W'.format(self.name),
								 regularizer=self.W_regularizer,
								 constraint=self.W_constraint)
		if self.bias:
			self.b = self.add_weight((input_shape[-1],),
									 initializer='zero',
									 name='{}_b'.format(self.name),
									 regularizer=self.b_regularizer,
									 constraint=self.b_constraint)

		self.u = self.add_weight((input_shape[-1],),
								 initializer=self.init,
								 name='{}_u'.format(self.name),
								 regularizer=self.u_regularizer,
								 constraint=self.u_constraint)

		super(AttentionWithContext, self).build(input_shape)

	def compute_mask(self, input, input_mask=None):
		# do not pass the mask to the next layers
		return None

	def call(self, x, mask=None):
		uit = dot_product(x, self.W)

		if self.bias:
			uit += self.b

		uit = K.tanh(uit)
		ait = dot_product(uit, self.u)

		a = K.exp(ait)

		# apply mask after the exp. will be re-normalized next
		if mask is not None:
			# Cast the mask to floatX to avoid float64 upcasting in theano
			a *= K.cast(mask, K.floatx())

		# in some cases especially in the early stages of training the sum may be almost zero
		# and this results in NaN's. A workaround is to add a very small positive number eps to the sum.
		# a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
		a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

		a = K.expand_dims(a)
		weighted_input = x * a
		return K.sum(weighted_input, axis=1)

	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[-1]


# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, cardinality):
	# generate random sequence
	sequence_in = generate_sequence(n_in, cardinality)
	sequence_out = [sequence_in[n_out-1], sequence_in[0]]  #+ [0 for _ in range(n_in-n_out)]
	# one hot encode
	X = one_hot_encode(sequence_in, cardinality)
	y = one_hot_encode(sequence_out, cardinality)
	# reshape as 3D
	X = X.reshape((1, X.shape[0], X.shape[1]))
	y = y.reshape((1, y.shape[0], y.shape[1]))
	return X,y

# define the encoder-decoder model
def baseline_model(n_timesteps_in, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features)))
	model.add(RepeatVector(n_timesteps_in))
	model.add(LSTM(150, return_sequences=True))
	model.add(TimeDistributed(Dense(n_features, activation='softmax')))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# define the encoder-decoder with attention model
def attention_model(n_timesteps_in, n_timesteps_out, n_features):
	model = Sequential()
	model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
	model.add(AttentionDecoder(150, n_features))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# define the encoder-decoder with attention model
def attention_model_new(n_timesteps_in, n_timesteps_out,n_features):
	model = Sequential()
	model.add(LSTM(200, input_shape=(n_timesteps_in, n_features), return_sequences=True))
	model.add(Attention())
	model.add(RepeatVector(n_timesteps_out))
	model.add(LSTM(output_dim=400, return_sequences=True)) 
	model.add(TimeDistributed(Dense(n_features)))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

def attention_model_recurrent_shop(n_timesteps_in, n_timesteps_out,n_features):
	model = AttentionSeq2Seq(input_dim=n_features, input_length=n_timesteps_in,  
			hidden_dim=100, output_length=n_timesteps_out, output_dim=n_features, depth=1)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
	return model

# train and evaluate a model, return accuracy
def train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features):
	# train LSTM
	for epoch in range(10000):
		# generate new random sequence
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		# fit model for one epoch on this sequence
		model.fit(X, y, epochs=1, verbose=0)
	# evaluate LSTM
	total, correct = 100, 0
	for _ in range(total):
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		yhat = model.predict(X, verbose=0)
		if array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
			correct += 1
	return float(correct)/float(total)*100.0


def runModel(n_features = 50, n_timesteps_in = 5, n_timesteps_out = 2, n_repeats = 2, runBaseline=False):
	print('Encoder-Decoder With Attention Model')
	results = list()
	for _ in range(n_repeats):
		model = attention_model(n_timesteps_in,n_timesteps_out, n_features)
		accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
		results.append(accuracy)
		print(accuracy)
	print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))

	for _ in range(10):
		X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
		yhat = model.predict(X, verbose=0)
		print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))

	return model
	if runBaseline:
		# evaluate encoder-decoder model
		print('Encoder-Decoder Model')
		results = list()
		for _ in range(n_repeats):
			model = baseline_model(n_timesteps_in, n_features)
			accuracy = train_evaluate_model(model, n_timesteps_in, n_timesteps_out, n_features)
			results.append(accuracy)
			print(accuracy)
		print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))
		# evaluate encoder-decoder with attention model
	