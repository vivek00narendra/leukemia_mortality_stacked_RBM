# Structure for RBM from PythonRepo.com
# https://pythonrepo.com/repo/echen-restricted-boltzmann-machines-python-deep-learning 

from __future__ import print_function
from sklearn import preprocessing
import numpy as np
import csv
import re

ds9 = csv.reader(open("ds9_class_cvs.csv"), delimiter=",")
patient_id = csv.reader(open("patientids.csv"), delimiter=",")
allaffy = csv.reader(open("allaffy_test.csv"), delimiter=",")

# Reading ds9 data
ds9_header = []
ds9_header = next(ds9)

ds9_rows = []
for row in ds9:
    ds9_rows.append(row)

# Reading allaffy data:
allaffy_header = []
allaffy_header = next(allaffy)

allaffy_rows = []
for row in allaffy:
    allaffy_rows.append(row)

# Reading patient ID data:
patient_id_rows = []
for row in patient_id:
    patient_id_rows.append(row)

allaffy_header.insert(0, 'patient_ID')
for i in range(len(patient_id_rows)):
    allaffy_rows[i].insert(0, patient_id_rows[i])


processed_allaffy = []

repeat_indices = [20, 45, 50, 55, 61, 70, 73, 76]
repeat_ids = [123.1, 293.1, 3.1, 308.1, 313.1, 326.1, 33.1, 338.1]

# Without Repeat patients

# Delete repeat values 
for i in repeat_indices:
    del allaffy_rows[i]

for i in range(len(allaffy_rows)):
    allaffy_rows[i][0] = allaffy_rows[i][0][0]
    temp = allaffy_rows[i]
    temp.pop(0)
    processed_allaffy.append([ds9_rows[i][0], ds9_rows[i][4], temp])

# for i in range(len(allaffy_rows)):
#     for j in range(len(ds9_rows)):
#         # if str(allaffy_rows[i][0][0].replace('.1', '')) == str((re.sub('\D', '', ds9_rows[j][0]))):
#         print(str((re.sub('\D', '', ds9_rows[j][0]))))
#         if str(allaffy_rows[i][0][0]) == str((re.sub('\D', '', ds9_rows[j][0]))):
#             allaffy_rows[i][0] = allaffy_rows[i][0][0]
#             temp = allaffy_rows[i]
#             temp.pop(0)
#             processed_allaffy.append([ds9_rows[j][0], ds9_rows[j][4], temp])

# for i in range(len(processed_allaffy)):
#     print(processed_allaffy[i][0])

# Changing 'y' and 'n' values to 1 and 0 for mortality
for i in range(len(processed_allaffy)):
  if processed_allaffy[i][1] == 'N':
    processed_allaffy[i][1] = 0 
  elif processed_allaffy[i][1] == 'Y':
    processed_allaffy[i][1] = 1

# Normalizing Values
for i in processed_allaffy:
    temp = np.array(i[2])
    normalized_temp = preprocessing.normalize([temp])
    normalized_temp = normalized_temp[0]
    normalized_temp = normalized_temp.tolist()
    i[2] = normalized_temp

# Training data and mortality values
training_data = []
mortality_data = []
for i in range(len(processed_allaffy)):
  training_data.append(processed_allaffy[i][2])

for i in range(len(processed_allaffy)):
  mortality_data.append(processed_allaffy[i][1])

training_data = np.array(training_data)
mortality_data = np.array(mortality_data)
print(np.shape(training_data))
print(np.shape(mortality_data))
    
class RBM:
  
  def __init__(self, num_visible, num_hidden):
    self.num_hidden = num_hidden
    self.num_visible = num_visible
    self.debug_print = True

    # Initialize a weight matrix, of dimensions 
    np_rng = np.random.RandomState(1234)

    self.weights = np.asarray(np_rng.uniform(
			low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                       	size=(num_visible, num_hidden)))


    # Insert weights for the bias units into the first row and first column.
    self.weights = np.insert(self.weights, 0, 0, axis = 0)
    self.weights = np.insert(self.weights, 0, 0, axis = 1)

  def train(self, data, max_epochs = 1000, learning_rate = 0.1):
    """
    Train the machine.

    data: A matrix where each row is a training example consisting of the states of visible units.    
    """

    num_examples = data.shape[0]

    # Insert bias units of 1 into the first column.
    data = np.insert(data, 0, 1, axis = 1)

    for epoch in range(max_epochs):      
      # "positive CD phase" aka the reality phase
      pos_hidden_activations = np.dot(data, self.weights)      
      pos_hidden_probs = self._logistic(pos_hidden_activations)
      pos_hidden_probs[:,0] = 1 # Fix the bias unit.
      pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
      pos_associations = np.dot(data.T, pos_hidden_probs)

      # "negative CD phase" aka the daydreaming phase
      neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
      neg_visible_probs = self._logistic(neg_visible_activations)
      neg_visible_probs[:,0] = 1 # Fix the bias unit.
      neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
      neg_hidden_probs = self._logistic(neg_hidden_activations)
      neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

      # Update weights.
      self.weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

      error = np.sum((data - neg_visible_probs) ** 2)
      if self.debug_print:
        print("Epoch %s: error is %s" % (epoch, error))

  def run_visible(self, data):
    """    
    Parameters
    ----------
    data: A matrix where each row consists of the states of the visible units.
    
    Returns
    -------
    hidden_states: A matrix where each row consists of the hidden units activated from the visible
    units in the data matrix passed in.
    """
    
    num_examples = data.shape[0]
    
    # Create a matrix, where each row is to be the hidden units (plus a bias unit)
    # sampled from a training example.
    hidden_states = np.ones((num_examples, self.num_hidden + 1))
    
    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the hidden units.
    hidden_activations = np.dot(data, self.weights)
    # Calculate the probabilities of turning the hidden units on.
    hidden_probs = self._logistic(hidden_activations)
    # Turn the hidden units on with their specified probabilities.
    hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
    # Always fix the bias unit to 1.
    # hidden_states[:,0] = 1
  
    # Ignore the bias units.
    hidden_states = hidden_states[:,1:]
    return hidden_states
    
  def run_hidden(self, data):
    """
    data: A matrix where each row consists of the states of the hidden units.
    Returns
    -------
    visible_states: A matrix where each row consists of the visible units activated from the hidden
    units in the data matrix passed in.
    """

    num_examples = data.shape[0]

    # Create a matrix, where each row is to be the visible units (plus a bias unit)
    # sampled from a training example.
    visible_states = np.ones((num_examples, self.num_visible + 1))

    # Insert bias units of 1 into the first column of data.
    data = np.insert(data, 0, 1, axis = 1)

    # Calculate the activations of the visible units.
    visible_activations = np.dot(data, self.weights.T)
    # Calculate the probabilities of turning the visible units on.
    visible_probs = self._logistic(visible_activations)
    # Turn the visible units on with their specified probabilities.
    visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
    # Always fix the bias unit to 1.
    # visible_states[:,0] = 1

    # Ignore the bias units.
    visible_states = visible_states[:,1:]
    return visible_states
    
  def daydream(self, num_samples):
    """
    Returns
    -------
    samples: A matrix, where each row is a sample of the visible units produced while the network was
    daydreaming.
    """

    # Create a matrix, where each row is to be a sample of of the visible units 
    # (with an extra bias unit), initialized to all ones.
    samples = np.ones((num_samples, self.num_visible + 1))

    # Take the first sample from a uniform distribution.
    samples[0,1:] = np.random.rand(self.num_visible)

    # Start the alternating Gibbs sampling.
    for i in range(1, num_samples):
      visible = samples[i-1,:]

      # Calculate the activations of the hidden units.
      hidden_activations = np.dot(visible, self.weights)      
      # Calculate the probabilities of turning the hidden units on.
      hidden_probs = self._logistic(hidden_activations)
      # Turn the hidden units on with their specified probabilities.
      hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
      # Always fix the bias unit to 1.
      hidden_states[0] = 1

      # Recalculate the probabilities that the visible units are on.
      visible_activations = np.dot(hidden_states, self.weights.T)
      visible_probs = self._logistic(visible_activations)
      visible_states = visible_probs > np.random.rand(self.num_visible + 1)
      samples[i,:] = visible_states

    # Ignore the bias units (the first column), since they're always set to 1.
    return samples[:,1:]        
      
  def _logistic(self, x):
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
  r = RBM(num_visible = 11288, num_hidden = 8192)
  training_data1 = np.array(training_data)
  print(training_data1.shape)
  # Point of diminishing returns is 90 epochs
  r.train(training_data1, max_epochs = 10)
  r.weights = np.delete(r.weights, 0, axis=0)
  r.weights = np.delete(r.weights, 0, axis=1)
  print(r.weights.shape)

  # Point of diminishing returns is 2 epochs
  r2 = RBM(num_visible = 8192, num_hidden = 4096)
  r2.train(r.weights, max_epochs = 10)
  print(r2.weights.shape)
  r2.weights = np.delete(r2.weights, 0, axis=0)
  r2.weights = np.delete(r2.weights, 0, axis=1)
  print(r2.weights.shape)
  
  # Point of diminishing returns is 9 epochs
  r3 = RBM(num_visible = 4096, num_hidden = 2048)
  r3.train(r2.weights, max_epochs = 10)
  print(r3.weights.shape)
  r3.weights = np.delete(r3.weights, 0, axis=0)
  r3.weights = np.delete(r3.weights, 0, axis=1)
  print(r3.weights.shape)

  # Point of diminishing returns is 46 epochs
  r4 = RBM(num_visible = 2048, num_hidden = 1024)
  r4.train(r3.weights, max_epochs = 10)
  print(r4.weights.shape)
  r4.weights = np.delete(r4.weights, 0, axis=0)
  r4.weights = np.delete(r4.weights, 0, axis=1)
  print(r4.weights.shape)

  # Point of diminishing returns is 7 epochs
  r5 = RBM(num_visible = 1024, num_hidden = 512)
  r5.train(r4.weights, max_epochs = 10)
  print(r5.weights.shape)
  r5.weights = np.delete(r5.weights, 0, axis=0)
  r5.weights = np.delete(r5.weights, 0, axis=1)
  print(r5.weights.shape)

  # Point of diminishing returns in 52 epochs
  r6 = RBM(num_visible = 512, num_hidden = 256)
  r6.train(r5.weights, max_epochs = 10)
  print(r6.weights.shape)
  r6.weights = np.delete(r6.weights, 0, axis=0)
  r6.weights = np.delete(r6.weights, 0, axis=1)
  print(r6.weights.shape)

  # Point of diminshing returns in 3 epochs.
  r7 = RBM(num_visible = 256, num_hidden = 149)
  r7.train(r6.weights, max_epochs = 10)
  print(r7.weights.shape)
  r7.weights = np.delete(r7.weights, 0, axis=0)
  r7.weights = np.delete(r7.weights, 0, axis=1)
  print(r7.weights.shape)
  print(r7.weights)

  final_weights = []
  for i in r7.weights:
    row_weights = []
    for j in i:
      norm_val = float(j)/sum(i)
      row_weights.append(norm_val)
    final_weights.append(row_weights)

  with open('allaffy_stacked_RBM_no_repeat.txt', 'w') as f:
    for i in final_weights:
      f.write(str(i))
      f.write('\n')

 
 
