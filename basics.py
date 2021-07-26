import tensorflow as tf
import numpy as np
import sklearn as sk


t = tf.zeros([5,5,5,5])
print(t)

t1 = tf.reshape(t,[125,-1]) # hen e do not kno the last number we put -1 (it is 5)
print(t1)


x1 = tf.constant(5)
x2 = tf.constant(6)
result = tf.multiply(x1,x2)
print(result)


string = tf.Variable("this is a string", tf.string) 
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
print(string)
print(number)
print(floating)


# Creating a 2D tensor
matrix = [[1,2,3,4,5],
          [6,7,8,9,10],
          [11,12,13,14,15],
          [16,17,18,19,20]]

tensor = tf.Variable(matrix, dtype=tf.int32) 
print(tf.rank(tensor))
print(tensor.shape)
print(



# Now lets select some different rows and columns from our tensor

three = tensor[0,2]  # selects the 3rd element from the 1st row
print(three)  # -> 3

row1 = tensor[0]  # selects the first row
print(row1)

column1 = tensor[:, 0]  # selects the first column
print(column1)

row_2_and_4 = tensor[1::2]  # selects second and fourth row
print(row_2_and_4)

column_1_in_row_2_and_3 = tensor[1:3, 0]
print(column_1_in_row_2_and_3)
