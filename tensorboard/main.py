import numpy as np
import tensorflow as tf

# placeholder-ийг python скриптээс Tensorflow графийн 
# үйлдлүүдрүү утга дамжуулахад ашигладаг
# оролтын feature x болон гаралтын y нарт зориулан хоёр placeholder үүсгэе
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# олохыг хүссэн функц маань 2-р зэргийн олон гишүүнт гэж үзье
# коффициентүүдийг нь хадгалах 3-н гишүүнтэй вектор үүсгэн хувиарлая
# энэ хувьсагч нь санамсаргүй утгуудаар автоматаар дүүрсэн байх болно
w = tf.get_variable("w", shape=[3, 1])

# yhat-ийг y-ийн дөхөлтөөр төлөөлүүлэн ашиглахын тулд тодорхойлоё
f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1)
yhat = tf.squeeze(tf.matmul(f, w), 1)

# loss-ийг y-ийн дөхөлт болон өөрийнх нь жинхэнэ утга хоёрын хоорондох l2 зай
# байхаар тодорхойлоё. 
loss = tf.nn.l2_loss(yhat - y) + 0.1 * tf.nn.l2_loss(w)

# Adam optimizer-ийг суралцах хэмжээ нь 0.1 тэйгээр loss-ийг 
# минимумчилахын тул тохируулан хэрэглэе
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

sess = tf.Session()
# Олон хувьсагчууд хэрэглэж байгаа болохоор эхлээд тэд нарыг цэнэглэх хэрэгтэй
sess.run(tf.global_variables_initializer())
for _ in range(1000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})
    print(loss_val)
print(sess.run([w]))

writer = tf.summary.FileWriter('./graphs', sess.graph)
writer.close()


