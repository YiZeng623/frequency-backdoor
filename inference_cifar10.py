import tensorflow as tf
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)
import argparse
import util
from keras.datasets import cifar10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--name', dest='name', required=True)
    args = parser.parse_args()

    #loading the original data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    #loding the pre-trained detection model
    model = util.cifar10_model()

    #prepare the dct2-transformed samples for test
    attackname = args.name
    x_dct_test,hot_test_lab = util.dct_test_data(x_test,attackname)


    all_loss, all_acc = model.evaluate(x_dct_test,hot_test_lab, batch_size=64)
    print('the loss over all the samples is {:.3f}, the acc is {:.3f}'.format(all_loss,all_acc))
    tri_loss, tri_acc = model.evaluate(x_dct_test[int(x_dct_test.shape[0]/2):],hot_test_lab[int(x_dct_test.shape[0]/2):], batch_size=64)
    print('the loss over triggered samples is {:.3f}, the acc is {:.3f}'.format(tri_loss,tri_acc))
