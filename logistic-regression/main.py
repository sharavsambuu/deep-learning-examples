from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# өгөгдлөө унших
mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data", one_hot=True)

# модель үүсгэх

# зургийн өгөгдөл 28x28 хэмжээтэй, None гэж тавьж өгснөөр
# хичнээнч олон тооны зураг хадгалах боломжтой гэсэн үг
x = tf.placeholder(tf.float32, [None, 784]) 
# зургийн хэмжээстэй таарч байхаас гадна гаралт нь 10-н ялгаатай
# утга бүхий байх тул weight матриц маань хэмжээсээ тооцон тохируулах ёстой
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))
# модель
y = tf.matmul(x, W) + b

# loss функцийг тодорхойлох
y_ = tf.placeholder(tf.float32, [None, 10])
# cross-entropy функц
# эдгээрийг хураангуйлсан tf.nn.softmax_cross_entropy_with_logits гэх функц бий
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# сургах
for _ in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# сургасан моделио шалгах
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
