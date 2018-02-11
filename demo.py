import caffe
import numpy as np
import cv2
import mvnc.mvncapi as mvnc

MODEL_DEF = './deploy.prototxt'
MODEL_WEIGHT = './duckie_model_iter_10000.caffemodel'
IMAGE_DIR = './sample_images/'
OMEGA_DIR = './sample_omega.csv'
GRAPH_PATH = './lala.graph'
omega = np.genfromtxt(OMEGA_DIR, delimiter=',', dtype=np.float32)
n_image = omega.shape[0]


def demo_on_computer():
    # this function predicts the 10 output by running caffe net on the computer
    net = caffe.Net(MODEL_DEF, MODEL_WEIGHT, caffe.TEST)
    for i in range(n_image):
        image = cv2.imread(IMAGE_DIR + '%05d.jpg' % (i + 1))
        # image.shape = [480 x 640 x 3]
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_NEAREST)
        # image.shape = [120 x 160 x 3]
        image = image[40:, :, :]
        # image.shape = [80 x 160 x 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('lala', image)
        cv2.waitKey(0)
        # image.shape = [80 x 160]
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # convert to [0,1]
        image = np.asarray(image)
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        net.blobs['data'].data[...] = image
        out = net.forward(end='out')
        print out['out'][0][0]


def demo_on_stick():
    # this function predicts the 10 outputs by loading a graph to the compute stick. All calculations are on the stick.
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()
    device = mvnc.Device(devices[0])
    device.OpenDevice()
    with open(GRAPH_PATH, mode='rb') as f:
        blob = f.read()
    graph = device.AllocateGraph(blob)
    for i in range(n_image):
        image = cv2.imread(IMAGE_DIR + '%05d.jpg' % (i + 1))
        # image.shape = [480 x 640 x 3]
        image = cv2.resize(image, (160, 120), interpolation=cv2.INTER_NEAREST)
        # image.shape = [120 x 160 x 3]
        image = image[40:, :, :]
        # image.shape = [80 x 160 x 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        image = np.expand_dims(image, axis=2)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        graph.LoadTensor(image.astype(np.float16), 'user object')
        output, userobj = graph.GetResult()
        print output[0]
    graph.DeallocateGraph()
    device.CloseDevice()


demo_on_computer()
demo_on_stick()
