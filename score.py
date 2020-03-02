from initModel import ftRankLoss
import numpy as np
import tensorflow as tf
import cv2
import argparse


# Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


def image_preprocess(img_path):
    pass


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('test_image_path', help='Path for test image.')
    args = parser.parse_args()

    # Get mean of AVA dataset
    mean_img = np.fromfile('./data/mean_AADB.txt', sep=',', dtype=np.float32)
    mean_img = np.reshape(mean_img, (256, 256, 3))

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))

    # Construct the network
    net = ftRankLoss({'imgLow': input_node})

    with tf.Session() as sess:
        # Load the model
        print('Loading the model...')
        net.load('./data/initModel.data', sess)

        # Load the image
        print('Loading the image...')
        img = cv2.imread(args.test_image_path)
        img = cv2.resize(img, (256, 256))
        img = img - mean_img
        img = cv2.resize(img, (227, 227))
        img = img.reshape((-1, 227, 227, 3))
        print(img.shape)

        # Perform a forward pass
        print('Scoring...')
        result = sess.run(net.get_output(), feed_dict={input_node: img})

        print(result)


if __name__ == '__main__':
    main()
