import numpy as np
import pandas as pd

# Question 1

#a
arr = np.array([1,2,3,6,4,5])
print(np.flip(arr))
# or simpyly
print(arr[::-1])

#b
arr = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
print(arr)
print(np.ravel(arr))
print(arr.flatten())

#c
# they are the same but are diff objects in memory
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])
print(arr1 == arr2) #eleement-wise comparison

#d 
x = np.array([1,2,3,4,5,1,2,1,1,1])
y = np.array(np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3, ]))

counts = np.bincount(x)
print("Val = ", np.argmax(counts),"at", np.where(x == np.argmax(counts))[0])

counts = np.bincount(y)
print("Val = ", np.argmax(counts),"at", np.where(y == np.argmax(counts))[0])

#e
gfg = np.matrix([[4, 1, 9],[12, 3, 1],[4, 5, 6]])
print(np.sum(gfg))
print(np.sum(gfg, axis = 1))
print(np.sum(gfg, axis = 0))

#f
np_array = np.array([[55, 25, 15],[30, 44, 2],[11, 45, 77]])
print(np.trace(np_array))

eig_val, eig_vec = np.linalg.eig(np_array)
print(eig_val)
print(eig_vec)
print(np.linalg.inv(np_array))
print(np.linalg.det(np_array))

#g

p = np.array([[1, 2], [2, 3]])
q = np.array([[4, 5], [6, 7]])
print(np.dot(p,q))
print(np.cov(p.T,q.T))

p2 = np.array([[1, 2], [2, 3], [4, 5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])
print("Product p2 x q2:\n", np.dot(p2, q2))
print("Covariance:\n", np.cov(p2[:2].T, q2.T))

#h
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])
print(np.inner(x,y))
print(np.outer(x.flatten(), y.flatten()))

from itertools import product
print(list(product(x,y)))

# Question 2

#a
#i
array = np.array([[1, -2, 3],[-4, 5, -6]])
print(np.abs(array))

#ii
percentile = [25, 50, 75]
print(np.percentile(np.ravel(array), percentile))
print(np.percentile(array, percentile, axis = 1))
print(np.percentile(array, percentile, axis = 0))

#iii
print(np.mean(np.ravel(array)))
print(np.median(np.ravel(array)))
print(np.std(np.ravel(array)))

print(np.mean(np.ravel(array)))
print(np.mean(np.ravel(array)))
print(np.mean(np.ravel(array)))

print(np.mean(array, axis = 1))
print(np.median(array, axis = 1))
print(np.std(array, axis = 1))

print(np.mean(array, axis = 0))
print(np.median(array, axis = 0))
print(np.std(array, axis = 0))

#b
a = np.array([-1.8, -1.6, -0.5, 0.5,1.6, 1.8, 3.0])
print(np.floor(a))
print(np.ceil(a))
print(np.trunc(a))
print(np.round(a))

# Question 3

#a
array = np.array([10, 52, 62, 16, 16, 54, 453])
print(np.sort(array))
print(np.argsort(array))
print(np.sort(array)[:4])
print(np.sort(array[-5:]))

#b
array = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
print(array[array == array.astype(int)])
print(array[array != array.astype(int)])

# Question 4

from PIL import Image
def img_to_array(path):
    img = Image.open(path)
    arr = np.array(img)
    print(arr)
    
    if len(arr.shape) == 2:
        np.savetxt("grayscale_image.txt", arr, fmt='%d')
        print("Saved as grayscale.")
    else:
        arr_reshaped = arr.reshape(-1, arr.shape[-1])
        np.savetxt("rgb_image.txt", arr_reshaped, fmt='%d')
        print("Saved as RGB.")
        

img_to_array("download.jpeg")

gray_data = np.loadtxt("grayscale_image.txt")
rgb_data = np.loadtxt("rgb_image.txt").reshape(-1, 3)
