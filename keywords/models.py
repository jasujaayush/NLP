import DataGenerator as dg
import wiki
from keras.models import Model
from keras.layers import Input, LSTM, Dense


batch_size = 64
epochs = 100 
latent_dim = 200 
num_samples = 10000
print "Initializing data generator object"
data_generator = dg.DataGenerator(wiki.getCmap())
vocab_size = data_generator.vocab_size
print "vocab_size : ", vocab_size

encoder_inputs = Input(shape=(None, vocab_size))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, vocab_size))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
print "Compiling model"
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model.fit([input_encoded_text, output_encoded_text], shifted_ouput_encoded_text,
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_split=0.2)

print "Fitting model"
training_generator = data_generator.generate()
model.fit_generator(generator = training_generator, steps_per_epoch = 40, epochs=epochs, validation_split=0.2, verbose = 1)
# Save model
model.save('s2s.h5')