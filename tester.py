# In[0]
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from skimage.transform import resize
import numpy as np
import pickle

# In[1]
img = cv2.imread('test/dist.jpg')
img = resize(img, (256, 256, 3), mode='constant', preserve_range=False)


# 690 in y axis ---> 71 inches

img = cv2.line(img,
         (0, 246),
         (256, 246),
         (255, 255, 0), 1)

# 604 in y axis ---> 94 inches
img = cv2.line(img,
         (0, 215),
         (256, 215),
         (255, 255, 0), 1)

# 512 in y axis ---> 141 inches
img = cv2.line(img,
         (0, 182),
         (256, 182),
         (255, 255, 0), 1)

# test_X in y axis ---> ?? To Find
test_X = 195
img = cv2.line(img,
         (0, test_X),
         (256, test_X),
         (255, 255, 0), 1)

plt.imshow(img)

x = np.array([[246], [215], [182]])
y = np.array([[71], [94], [141]])

print(x.shape)
model = LinearRegression()
model.fit(x, y)

# save
with open('weights/distance_model.pkl','wb') as file:
    pickle.dump(model, file)

# load
with open('weights/distance_model.pkl', 'rb') as file:
    dist_model = pickle.load(file)
print(dist_model.predict([[test_X]]))
plt.show()