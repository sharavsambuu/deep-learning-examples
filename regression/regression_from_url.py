import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import requests

# өгөгдлүүдээ http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr05.html 
# хаяг дээрхи Чикаго хотын гал болон гэмт хэргийн гаралтын хамаарлын 
# судалгааг агуулсан excel файлаас татаж авч уншаад numpy массив болгож хадгална
data_url = "http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/excel/slr05.xls"
u = requests.get(data_url)
book = xlrd.open_workbook(file_contents=u.content, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows-1

# гал гаралтын тоо тэмжээг илтгэх оролтын X placeholder
X = tf.placeholder(tf.float32, name="X")
# гэмт хэргийн гаралтын тоо хэмжээг илтгэх гаралтын Y placeholder
Y = tf.placeholder(tf.float32, name="Y")

# 0 утгаар цэнэглэсэн жин болон биас утгууд
w = tf.Variable(0., name="weights")
b = tf.Variable(0., name="bias")

# гал гаралтын тооноос гэмт хэргийн гаралтыг тооцон олох
# шугам регрессийн модел
Y_predicted = X*w+b

# алдааны квадрат утгыг төлөөлөх loss функц
loss = tf.square(Y-Y_predicted, name='loss')

# gradient descent алгоритм хэрэглэн 0.01 хэмжээтэйгээр суралцуулж
# loss утгыг минимумчилна
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# регрессийн цэгүүдийг зурж харуулах
plt.xlabel('Гал гаралтын тоо хэмжээ')
plt.ylabel('Гэмт хэргийн гаралтын тоо')
plt.scatter(data[:, 0], data[:, 1])
x_plot = np.linspace(0, 50, 100)
plt.ion() # графикийг тусдаа процесстой цонх болгон харуулах

with tf.Session() as sess:
    # шаардлагатай хувьсагчуудыг цэнэглэн зарлах, энэ тохиолдолд w болон b 
    sess.run(tf.global_variables_initializer())
    # моделийг сургах
    for i in range(100): # 100 epoch
        for x, y in data:
            # loss-ийг минимумчлах train_op
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # w болон b утгууд нь одоо ямар болсныг авах
        w_value, b_value = sess.run([w, b])
        print("weight=", w_value, ", bias=", b_value)

        # регрессийн шулууныг w болон b утгуудыг ашиглан зурах
        plt.xlabel('Гал гаралтын тоо хэмжээ')
        plt.ylabel('Гэмт хэргийн гаралтын тоо')
        plt.scatter(data[:, 0], data[:, 1])
        plt.plot(x_plot, x_plot*w_value + b_value)
        plt.show()
        plt.pause(0.01)
        plt.gcf().clear()


