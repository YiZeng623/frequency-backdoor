# Python
Python code to train the frequency-based backdoor data detector using the CIFAR-10 dataset as the example.

## Usage
A pre-trained detector model, '6_CNN_CIFAR10.h5py', is presented under the 'detector' folder. One can directely run the provided notebook, 'Train_Detection.ipynb', to test the pre-trained model on dct processed triggers.
Training a model from schrach can also be achieved using the code provided in the notebook.

All the evaluated triggers are stored under the 'trigger' folder. Including a smooth trigger generated based on the algorithm proposed
in the Section 5. The provided raw model does not work well with the smooth trigger as the analysis provided in the paper. However, one can fine-tune the model follow
the pipline provided in the paper to attain the results shown in the paper. We provide a fine-tuned weights as 'Tuned_CIFAR10.h5py', one can test the results of it on all the evaluated triggers.

Feel free to try some new triggers with this frequency-based etecor and enjoy!


