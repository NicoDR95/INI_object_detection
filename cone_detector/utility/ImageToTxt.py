import numpy as np
import cv2

image = '/home/nico/Desktop/400442.jpg'
image = cv2.imread(image)
image = cv2.resize(image, (512, 256))
# cv2.imshow('im', image)
# cv2.waitKey()
# image = np.reshape(image, newshape=(3, 800, 1600))
print(image.shape)
print(image)

b, g, r = cv2.split(image)

image = cv2.merge((b, g, r))
# cv2.imshow('im', image)
# cv2.waitKey()
print(b.shape)
# image = np.reshape(image, newshape=(800, 1600, 3))

# b = np.arange(10).reshape((2,5))
# g = np.arange(10).reshape((2,5))
# r = np.arange(10).reshape((2,5))



with open('/home/nico/Desktop/test_image.txt', 'w') as outfile:
    # for channel in image:
    np.savetxt(fname=outfile, X=r, delimiter="\n", fmt="%i")
    np.savetxt(fname=outfile, X=r, delimiter="\n", fmt="%i")
    np.savetxt(fname=outfile, X=r, delimiter="\n", fmt="%i")

    # np.savetxt(outfile, r)
    # np.savetxt(outfile, g)
    # np.savetxt(outfile, b)




# read_image = np.loadtxt('/home/nico/Desktop/test_image.txt')
# print(read_image.shape)
# np.reshape(read_image, newshape=(800, 1600, 3))
# read_image = np.reshape(read_image, newshape=(800, 1600, 3))


# cv2.imshow('im', read_image)
# cv2.waitKey()
# np.savetxt('/home/nico/Desktop/test_image.txt', image)
