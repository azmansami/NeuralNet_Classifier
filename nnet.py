import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import math


print("Tensorflow version: "+tf.__version__)
tf.set_random_seed(123)


def getLearningRate(i,decay=True):
    max_learning_rate=0.003
    if not decay:
        return max_learning_rate
    min_learning_rate=0.0001
    decay_speed=2000.0
    lr=min_learning_rate+(max_learning_rate-min_learning_rate)*math.exp(-i/decay_speed)
    return lr

# dropout parameter, probKeep 1 means all the weights are updated.
# probKeep < 1 means that only probKeep percentile of weights are updated. rest
# are frozen. A regularization implementation for neural net.
probKeep=0.75

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

x=tf.placeholder(tf.float32, shape=[None,28,28,1],name="input")
y_=tf.placeholder(tf.float32, shape=[None,10],name="y_actual")

lr=tf.placeholder(tf.float32,name="LearningRate")
tf.summary.scalar('LearnRate',lr)

pkeep=tf.placeholder(tf.float32,name="Regularization")
# 5 layer neural net.
# layer 1 = 200 node
# layer 2 = 100 node
# layer 3 =  60 node
# layer 4 =  30 node

L1=200
L2=100
L3=60
L4=30


#the model

xx=tf.reshape(x,[-1,784],name='Input_layer')

with tf.name_scope('Hidden_layer1'):
    w1 = tf.Variable(tf.truncated_normal([784,L1],stddev=0.1),name="W")
    b1 = tf.Variable(tf.zeros([L1])/10,name="B")
    y1=tf.nn.relu(tf.matmul(xx,w1)+b1)
    y1d=tf.nn.dropout(y1,pkeep)

with tf.name_scope('Hidden_layer2'):
    w2 = tf.Variable(tf.truncated_normal([L1,L2],stddev=0.1),name="W")
    b2 = tf.Variable(tf.zeros([L2])/10,name="B")
    y2=tf.nn.relu(tf.matmul(y1d,w2)+b2)
    y2d=tf.nn.dropout(y2,pkeep)

with tf.name_scope('Hidden_layer3'):
    w3 = tf.Variable(tf.truncated_normal([L2,L3],stddev=0.1),name="W")
    b3 = tf.Variable(tf.zeros([L3])/10,name="B")
    y3=tf.nn.relu(tf.matmul(y2d,w3)+b3)
    y3d=tf.nn.dropout(y3,pkeep)

with tf.name_scope('Hidden_layer4'):
    w4 = tf.Variable(tf.truncated_normal([L3,L4],stddev=0.1),name="W")
    b4 = tf.Variable(tf.zeros([L4])/10,name="B")
    y4=tf.nn.relu(tf.matmul(y3d,w4)+b4)
    y4d=tf.nn.dropout(y4,pkeep)

with tf.name_scope('Hidden_layer5'):
    w5 = tf.Variable(tf.truncated_normal([L4,10],stddev=0.1),name="W")
    b5 = tf.Variable(tf.zeros([10]),name="B")
    lin_pred=tf.matmul(y4d,w5)+b5

with tf.name_scope('Output_layer'):
    y=tf.nn.softmax(lin_pred)

#loss
with tf.name_scope("loss"):
    #As per documentation, this function takes linear_predictors and does
    #softmax conversion internally for efficiency.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=lin_pred, labels=y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
tf.summary.scalar('Xentropy',cross_entropy)

# accuracy
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
tf.summary.scalar('Accuracy',accuracy)

with tf.name_scope("train"):
    train=tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy)

# initialize tensorflow artifacts
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#setting up the filewirters
merged_summary=tf.summary.merge_all()
train_writer=tf.summary.FileWriter("/tmp/tboard/nnet/train")
test_writer=tf.summary.FileWriter("/tmp/tboard/nnet/test")
train_writer.add_graph(sess.graph)


for i in range(10000):
    learning_rate=getLearningRate(i,True)

    batch_x,batch_y=mnist.train.next_batch(100)
    train_summary,_,_=sess.run([merged_summary,train,accuracy],feed_dict={x:batch_x,y_:batch_y,
                                                                          lr:learning_rate,pkeep:probKeep})
    test_summary,test_acc=sess.run([merged_summary,accuracy],feed_dict={x:mnist.test.images,
                                                                        y_:mnist.test.labels,
                                                                        lr:learning_rate,pkeep:1.0})
    if i % 50 == 0:
        print("step: ",i,"learning rate: ","%0.5f" % learning_rate," test accuracy:",test_acc)
        train_writer.add_summary(train_summary,i)
        test_writer.add_summary(test_summary,i)
