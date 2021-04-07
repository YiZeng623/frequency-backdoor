import numpy as np
from deepfool import tar_deepfool
from gauss_smooth import gauss_smooth,normalization,smooth_clip
import os

def universal_perturbation(dataset, f, grads, target, delta=0.2, max_iter_uni = np.inf, num_classes=10, overshoot=0.02, max_iter_df=200):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)

    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).

    :param grads: gradient functions with respect to input (as many gradients as classes).

    :param delta: controls the desired fooling rate (default = 80% fooling rate)

    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)

    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)

    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).

    :param max_iter_df: maximum number of iterations for deepfool (default = 10)

    :return: the universal perturbation.
    """

    v = np.zeros_like((32,32,3))
    best_frate = 0.0
    fooling_rate = 0.0
    num_images =  np.shape(dataset)[0] # The images should be stacked ALONG FIRST DIMENSION
    file_perturbation = os.path.join('data', 'best_universal.npy')

    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni:
        # Shuffle the dataset
        np.random.shuffle(dataset)

        print ('Starting pass number ', itr)

        # Go through the data set and compute the perturbation increments sequentially
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :, :]

            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img+v)).flatten())):

                # Compute adversarial perturbation
                dr, iter, _, _ = tar_deepfool(cur_img + v, f, grads, target=target, num_classes=num_classes,
                                              overshoot=overshoot, max_iter=max_iter_df)
                # Make sure it converged...
                if iter < max_iter_df-1:
                    assert not np.any(np.isnan(dr))
                    assert np.all(np.isfinite(dr))

                    v = v + gauss_smooth(dr)
                    v = gauss_smooth(v)
                    v = normalization(v)


        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))

        # Yi: update the target label
        dif_count = est_labels_pert[np.where(est_labels_pert != est_labels_orig)]
        if len(dif_count)>(dataset.shape[0]*0.05):
            counts = np.bincount(dif_count.astype(np.int))
            target = np.argmax(counts)


        print(dif_count)
        print('the dominant label is:', target)
        print('FOOLING RATE = ', fooling_rate)
        if fooling_rate >= best_frate:
            best_v = v
            best_frate = fooling_rate
            new_target = target
            print('the best fooling rate updating to:',best_frate)
            print('the target label is updating to:', new_target)
            np.save(os.path.join(file_perturbation), best_v)

    return best_v,new_target
