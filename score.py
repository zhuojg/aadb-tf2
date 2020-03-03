from initModel import ftRankLoss
import numpy as np
import cv2
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Score:
    def __init__(self, model_data_path, mean_data_path):
        # Get mean of AVA dataset
        self.mean_img = np.fromfile(mean_data_path, sep=',', dtype=np.float32)
        self.mean_img = np.reshape(self.mean_img, (256, 256, 3))

        # Create a placeholder for the input image
        self.input_node = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))

        # Construct the network
        self.net = ftRankLoss({'imgLow': self.input_node})
        self.sess = tf.Session()

        self.net.load(model_data_path, self.sess)

    def score_one_image(self, img):
        img = cv2.resize(img, (256, 256))
        img = img - self.mean_img
        img = cv2.resize(img, (227, 227))
        img = img.reshape((-1, 227, 227, 3))

        result = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})

        return result


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

        # Perform a forward pass
        print('Scoring...')
        result = sess.run(net.get_output(), feed_dict={input_node: img})

        print(result)


if __name__ == '__main__':
    main()
