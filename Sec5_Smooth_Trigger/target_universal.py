import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

K.set_learning_phase(0)

from universal_pert import universal_perturbation
device = '/gpu:0'
num_classes = 10

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
         tf.constant(0, tf.int32),
         tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

target = 6 #You can also use a random label to let the model find its own dominant label

with tf.device(device):
    persisted_sess = tf.Session()
    inception_model_path = os.path.join('model', 'cifar10_model.pb')

    model = os.path.join(inception_model_path)

    # Load the Inception model
    with gfile.FastGFile(model, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    persisted_sess.graph.get_operations()

    persisted_input = persisted_sess.graph.get_tensor_by_name("conv2d_18_input:0")
    persisted_output = persisted_sess.graph.get_tensor_by_name("dense/Softmax:0")

    print('>> Computing feedforward function...')
    def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 32, 32, 3))})

    file_perturbation = os.path.join('data', 'best_universal.npy')

    if os.path.isfile(file_perturbation) == 0:

        # TODO: Optimize this construction part!
        print('>> Compiling the gradient tensorflow functions. This might take some time...')
        y_flat = tf.reshape(persisted_output, (-1,))
        inds = tf.placeholder(tf.int32, shape=(num_classes,))
        dydx = jacobian(y_flat,persisted_input,inds)

        print('>> Computing gradient function...')
        def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

        # Load training data
        X = x_train[:100]

        # Running universal perturbation
        v,new_target = universal_perturbation(X, f, grad_fs, target=target, delta=0.8, max_iter_uni=np.inf, num_classes=num_classes, overshoot=0.02)

        # Saving the universal perturbation
        print('the final target label is:', new_target)
        np.save(os.path.join(file_perturbation), v)

    else:
        print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
        v = np.load(file_perturbation)

    print('>> Testing the universal perturbation on an image')


    image_original = x_test[0:1]
    label_original = np.argmax(f(image_original), axis=1)
    str_label_original = classes[np.int(label_original)]

    # Clip the perturbation to make sure images fit in uint8
    clipped_v = np.clip(image_original[0,:,:,:]+v[0,:,:,:], 0, 1) - image_original[0,:,:,:]

    image_perturbed = image_original + clipped_v[None, :, :, :]
    label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
    str_label_perturbed = classes[np.int(label_perturbed)]

    # Show original and perturbed image
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image_original[0, :, :, :].astype(dtype='float'), interpolation=None)
    plt.title(str_label_original)

    plt.subplot(1, 2, 2)
    plt.imshow(image_perturbed[0, :, :, :].astype(dtype='float'), interpolation=None)
    plt.title(str_label_perturbed)

    plt.show()