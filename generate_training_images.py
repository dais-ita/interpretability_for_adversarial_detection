
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

path_list = str(os.path.abspath(__file__)).split("/")

add_path = os.path.join(*path_list[1:-2])

# import sys
# sys.path.insert(0, '../')
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances, produce_relevance_image
import modules.render as render
import input_data

import tensorflow as tf
import numpy as np
import pdb
import scipy.io as sio

import foolbox

from foolbox.models import TensorFlowModel
from foolbox.criteria import TargetClassProbability
from foolbox.attacks import LBFGSAttack, DeepFoolAttack, GradientAttack, BoundaryAttack

from PIL import Image


print(foolbox.__file__)

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 1,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 1000,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 500,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_convolutional_logs','Summaries directory')
flags.DEFINE_boolean("relevance", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/epsilon/ww/flat/alphabeta')
flags.DEFINE_boolean("save_model", False,'Save the trained model')
flags.DEFINE_boolean("reload_model", False,'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", 'mnist_convolutional_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", 'mnist_convolutional_model','Checkpoint dir')

FLAGS = flags.FLAGS


def nn():
    
    return Sequential([Convolution(output_depth=10,input_depth=1,batch_size=FLAGS.batch_size, input_dim=28, act ='relu', stride_size=1, pad='VALID'),
                       AvgPool(),

                       Convolution(output_depth=25,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=4,output_depth=100,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                       
    ])





def feed_dict(mnist, train):
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        k = 1.0

    xs = xs.reshape(FLAGS.batch_size,28,28,1)
    return (2*xs)-1, ys, k


mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True



with tf.Session(config=config) as sess:
    ### Input tensorflow placeholders
    x = tf.placeholder(tf.float32, [None, 28,28,1], name='absolute_input_x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='absolute_output_y')
    keep_prob = tf.placeholder(tf.float32)

    ### network layers and output layver variable
    net = nn()
    inp = tf.pad(tf.reshape(x, [1,28,28,1]), [[0,0],[2,2],[2,2],[0,0]], name='absolute_input')
    op = net.forward(inp)
    y = tf.squeeze(op, name='absolute_output')
    
    ### relevancy map set up
    LRP = net.lrp(op, FLAGS.relevance_method, 1e-3)

    relevance_layerwise = []
    R = op
    for layer in net.modules[::-1]:
        R = net.lrp_layerwise(layer, R, FLAGS.relevance_method, 1e-3)
        relevance_layerwise.append(R)

    
    tf.global_variables_initializer().run()
    
    ### Start the main program flow

    ### load from pretrained model MNIST model
    tvars = tf.trainable_variables()
    npy_files = np.load('mnist_convolutional_model/model.npy', encoding='bytes')
    [sess.run(tv.assign(npy_files[tt])) for tt,tv in enumerate(tvars)]
    


    d = feed_dict(mnist, False) #load the mnist data
    
    print(x)
    print(y)
    ### initialisation of the adversarial attack variables
    model = TensorFlowModel(x, y, bounds=(-1, 1)) #model used by the attacks

    attacks = [] 

    #attacks.append(("LBFGSAttack", LBFGSAttack(model)))
    attacks.append(("DeepFoolAttack", DeepFoolAttack(model)))
    # attacks.append(("BoundaryAttack", BoundaryAttack(model))) # this attack currently doesn't work
    attacks.append(("GradientAttack", GradientAttack(model)))
    

    ### set up the output path
    training_data_path = "training_data"
    if not os.path.exists(training_data_path):
                os.mkdir(training_data_path);


    for attack_t in attacks: #generate data for each attack
        print("Generating data for: "+str(attack_t[0]))

        ### set up of output paths for this attack 
        attack_path = os.path.join("training_data",attack_t[0])

        raw_image_path = os.path.join(attack_path,"input_images")
        rel_image_path = os.path.join(attack_path,"rel_images")

        image_dir = os.path.join(raw_image_path,"0")
        adv_image_dir = os.path.join(raw_image_path,"1")
        rel_image_dir = os.path.join(rel_image_path,"0")
        adv_rel_image_dir = os.path.join(rel_image_path,"1")

        check_paths = [attack_path,raw_image_path,rel_image_path,image_dir,adv_image_dir,rel_image_dir,adv_rel_image_dir]
        for check_path in check_paths:
            if not os.path.exists(check_path):
                os.mkdir(check_path);

        attack = attack_t[1] #move the attack instance from the attack tuple to a variable named attack (for legacy reasons)

        no_adversarial_list = [] #list of images that an adversarial attack could not be generated for

        ground_truths = [] #contains the training data to be output to a csv file

        for i in range(len(d[0][:])): #for each MNIST image in the batch loaded in d[0]

            ###fetch and save the image from MNIST
            image = d[0][i]

            save_im = image.reshape(28,28)
            save_im = (255.0 / save_im.max() * (save_im - save_im.min())).astype(np.uint8)
            im = Image.fromarray(save_im)
           

            label = np.argmax(model.batch_predictions([image])) #get the class prediciton from the model

            im_output_path = os.path.join(image_dir,"test_"+str(i)+"_"+str(label)+".jpeg")
            im.save(im_output_path)



            ### create and save the adversarial image
            adversarial = attack(image, label=label) 

            if(adversarial is None):
                print("no adversarial attack found")
                no_adversarial_list.append(i)
                continue

            save_im = adversarial.reshape(28,28)
            save_im = (255.0 / save_im.max() * (save_im - save_im.min())).astype(np.uint8)
            im = Image.fromarray(save_im)
            
            adv_label = np.argmax(model.batch_predictions([adversarial])) #get the class prediction for the adversarial image
            
            adv_im_output_path = os.path.join(adv_image_dir,"adv_test_"+str(i)+"_"+str(label)+"_"+str(adv_label)+".jpeg")
            im.save(adv_im_output_path)

            

            ### relevancy map generation and save
            rel_inp = image.reshape(1,28,28,1) # reshape to how tensorflow expects

            test_inp = {x:rel_inp, y_: d[1][i:i+1], keep_prob: d[2]} #form feed dict
            y1, relevance_test, rel_layer= sess.run([y, LRP, relevance_layerwise], feed_dict=test_inp) #make class prediciton
            relevance_test = relevance_test[:,2:30,2:30,:]
            images = test_inp[x]
            relevance_image = produce_relevance_image(relevance_test.reshape([1,28,28,1]) )
            
            save_im = relevance_image.reshape(28,28,3)
            save_im = (255.0 / save_im.max() * (save_im - save_im.min())).astype(np.uint8)
            im = Image.fromarray(save_im, 'RGB')
            
            rel_im_output_path = os.path.join(rel_image_dir,"rel_test_"+str(i)+"_"+str(label)+"_"+str(adv_label)+".jpeg")
            im.save(rel_im_output_path)

            ### adversarial relevancy map generation and save 
            adv_inp = adversarial.reshape(1,28,28,1)

            test_inp = {x:adv_inp, y_: d[1][i:i+1], keep_prob: d[2]}
            y1, relevance_test, rel_layer= sess.run([y, LRP, relevance_layerwise], feed_dict=test_inp)
            relevance_test = relevance_test[:,2:30,2:30,:]
            images = test_inp[x]
            relevance_image = produce_relevance_image(relevance_test.reshape([1,28,28,1]) )
            
            save_im = relevance_image.reshape(28,28,3)
            save_im = (255.0 / save_im.max() * (save_im - save_im.min())).astype(np.uint8)
            im = Image.fromarray(save_im, 'RGB')
            
            adv_rel_im_output_path = os.path.join(adv_rel_image_dir,"adv_rel_test_"+str(i)+"_"+str(label)+"_"+str(adv_label)+".jpeg")
            im.save(adv_rel_im_output_path)
            
            print("original label: "+str(label))
            print("adversarial label: "+str(adv_label))

            ground_truths.append( (str(i),str(im_output_path),str(adv_im_output_path),str(rel_im_output_path),str(adv_rel_im_output_path),str(label),str(adv_label)) )

        print("Num of no adversarials:")
        print(len(no_adversarial_list))

        ### output training data to CSV
        ground_truth_output_string = "i,img_path,adv_img_path,rel_path,adv_rel_path,label,adv_label\n"

        for gt in ground_truths:
            ground_truth_output_string += gt[0] + "," + gt[1] + "," + gt[2] + "," + gt[3] + "," + gt[4] + "," + gt[5]+ "," + gt[6]+ "\n"

        ground_truth_output_path = os.path.join("training_data",attack_t[0]+"_adv_gt.csv")

        with open(ground_truth_output_path, "w") as f:
            f.write(ground_truth_output_string)




