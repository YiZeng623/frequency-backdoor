# Python
Python code to find a Smooth Trigger can lure 20ish percent of the clean sample to be misclassified using the pre-trained model.
The code evaluates the dominant label that other classes tend to be misclassified as. This dominant label would be the target label for the backdoor attack.
We are combining the found trigger with the dominant label to achieve the full pipeline of the low-frequency backdoor attack evaluated in the paper. 
Some of the code and the big-structure were referring to the Universal Adversarial Purturbation[1].

## Usage
One can directly run the 'target_universal.py' to observe the raw effect of the smooth trigger over the test data. 
It should demonstrates the original label and the the label after patching the trigger. The dominante label is frog which is the results in our end.

### Computing a universal perturbation for Cifar10 based on a pre-trained model
One can delete the 'best_universal.npy' file stored under the 'data' folder and rerun the 'target_universal.py' to compute a new smooth trigger acoording to the 
pretrained model and 100 samples from the trainig set. The pre-set delta is 0.8, as the model will output a trigger pattern which causes 20% clean sample be misclassified after patching. 


## Reference
[1] S. Moosavi-Dezfooli\*, A. Fawzi\*, O. Fawzi, P. Frossard:
[*Universal adversarial perturbations*](http://arxiv.org/pdf/1610.08401), CVPR 2017.
