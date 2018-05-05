# interpretability_for_adversarial_detection
Work exploring the use of interpretability techniques such as saliency maps to help detect machine learning adversarial attacks

Training data generation code is for python3

After installing python module requirements, place the files found in 'foolbox_replacement_files/models' in to 'foolbox/models' in your site-packages directory. (Due to this, the use of a virtual enviroment is reccomended)

Run cifar_util.py using python2 (due to the way the cifar images were 'pickled') to produce the cifar_10 images used to produce the adversarial detector training data. 

mnist training data generation: generate_training_images.py
cifar_10 training data generation: cifar_generate_training_images.py
