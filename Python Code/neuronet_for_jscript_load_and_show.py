from keras.datasets import mnist # subroutines for fetching the MNIST dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Dense # the two types of neural network layer we will be using
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

from keras.models import load_model

model = load_model('c:/users/anton.tuflin/trained_model_for_jscript.h5')
jcode=""
weights_0 = model.layers[0].get_weights()
line_0 = weights_0[0].transpose()
arr1=""
for col in line_0:
    arr2 = ""
    for line in col:
        arr2 =  arr2 + "," + str(line) 
    arr1= arr1 + "[" + arr2 + "],\n"
jcode =jcode + "var w12 = [" + arr1 + "];\n"

line_1 = weights_0[1]
arr1=""
for line in line_1:
    arr1= arr1  + "," +str(line)
jcode = jcode + "var bias2 = [" + arr1 + "]\n"


weights_1 = model.layers[2].get_weights()
line_0 = weights_1[0].transpose()
arr1=""
for col in line_0:
    arr2 = ""
    for line in col:
        arr2 =  arr2 + "," + str(line) 
    arr1= arr1 + "[" + arr2 + "],\n";
jcode = jcode + "var w23 = [" + arr1 + "];\n"

line_1 = weights_1[1]
arr1=""
for line in line_1:
    arr1= arr1  + "," +str(line)
jcode = jcode + "var bias3 = [" + arr1 + "]\n"

jcode=jcode.replace(",]","]")
jcode=jcode.replace(",\n]","\n]")
jcode=jcode.replace("[,","[")

print(jcode)
