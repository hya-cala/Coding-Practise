'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

# Data specific constants
n_inputs = 784
n_classes = 10

# Hyperparameters 
max_epoch = 12000
learning_rate = 0.5
seed = 0
n_hidden = 32
batch_size = 10

# Simoid prime
def sigmaprime(x):
    return tf.multiply(tf.sigmoid(x),tf.subtract(tf.constant(1.0),tf.sigmoid(x)))

# Build the model
def Multilayer_perceptron(x,weights,biases):
    #Hidden layer with sigmoid activation
    h_layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
    out_layer_1 = tf.sigmoid(h_layer_1)
    
    #Output layer with sigmoid activation
    output_layer = tf.matmul(out_layer_1,weights['out']) +biases['out']
    out_layer = tf.sigmoid(output_layer)
    
    return out_layer, output_layer, out_layer_1,h_layer_1

# Define the weights and biases
weights = {'h1': tf.Variable(tf.random_normal([n_inputs,n_hidden],seed= seed)),
           'out': tf.Variable(tf.random_normal([n_hidden,n_classes],seed = seed))}
biases = {'h1': tf.Variable(tf.random_normal([1,n_hidden],seed= seed)),
           'out': tf.Variable(tf.random_normal([1,n_classes],seed = seed))}

# Build the Graph
x_in = tf.placeholder(tf.float32,[None,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])

#Forward Pass
y_hat,h_2,o_1,h_1 = Multilayer_perceptron(x_in,weights,biases)

err = y_hat-y

#Backward Pass
delta_2 = tf.multiply(err, sigmaprime(h_2))
delta_w_2 = tf.matmul(tf.transpose(o_1),delta_2)

wtd_error = tf.matmul(delta_2,tf.transpose(weights['out']))
delta_1 = tf.multiply(wtd_error, sigmaprime(h_1))
delta_w_1 = tf.matmul(tf.transpose(x_in),delta_1)

# update weights
eta = tf.constant(learning_rate)
step = [tf.assign(weights['h1'],tf.subtract(weights['h1'],tf.multiply(eta,delta_w_1))),
        tf.assign(biases['h1'],tf.subtract(biases['h1'],tf.multiply(eta,tf.reduce_mean(delta_1,axis = [0])))),
        tf.assign(weights['out'],tf.subtract(weights['out'],tf.multiply(eta,delta_w_2))),
        tf.assign(biases['out'],tf.subtract(biases['out'],tf.multiply(eta,tf.reduce_mean(delta_2,axis = [0]))))]

# Define Accuracy
acct_mat = tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))
accuracy = tf.reduce_sum(tf.cast(acct_mat,tf.float32))

# Initiate the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(max_epoch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(step,feed_dict = {x_in:batch_xs,y:batch_ys})
        if epoch%1000 == 0:
            acc_test = sess.run(accuracy,feed_dict = {x_in:mnist.test.images,y:mnist.test.labels})
            acc_train = sess.run(accuracy,feed_dict = {x_in:mnist.train.images,y:mnist.train.labels})
            print('Epoch:{0} Accuracy Train%: {1} Accuracy Test%: {2}'.format(epoch,acc_train/600,acc_test/100))
    print('Epoch:{0} Accuracy Train%: {1} Accuracy Test%: {2}'.format(epoch,acc_train/600,acc_test/100))
'''

from keras.datasets import mnist 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam


(x_train,y_train),(x_test,y_test) = mnist.load_data()

n_hidden = 32
batch_size = 10
n_classes = 10
n_epoch = 2
imgsize = 784

x_train = x_train.reshape(-1,imgsize).astype('float32')/255
x_test = x_test.reshape(-1,imgsize).astype('float32')/255

y_train = np_utils.to_categorical(y_train,n_classes)
y_test = np_utils.to_categorical(y_test,n_classes)

model = Sequential([Dense(n_hidden,input_shape = (imgsize,),activation = 'sigmoid'),
                    Dense(n_classes,input_shape = (n_hidden,),activation = 'sigmoid')])

model.summary()
model.compile(optimizer = Adam(),loss = 'mse',metrics = ['accuracy'])

model.fit(x_train,y_train,batch_size = 10,epochs = n_epoch,validation_data = (x_test,y_test))

score = model.evaluate(x_test,y_test)
print(score)




































            
    






