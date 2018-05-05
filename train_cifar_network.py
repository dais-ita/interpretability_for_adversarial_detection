'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin 
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import modules.render as render
import input_data

import tensorflow as tf
import numpy as np
import pdb
import scipy.io as sio


from cifar_util import LoadCifarDataFromImages,LoadGSCifarData 

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("max_steps", 20000,'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 1000,'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 500,'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01,'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", 'data','Directory for storing data')
flags.DEFINE_string("summaries_dir", 'mnist_convolutional_logs','Summaries directory')
flags.DEFINE_boolean("relevance", False,'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/epsilon/ww/flat/alphabeta')
flags.DEFINE_boolean("save_model", True,'Save the trained model')
flags.DEFINE_boolean("reload_model", True,'Restore the trained model')
#flags.DEFINE_string("checkpoint_dir", 'mnist_convolution_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_dir", 'cifar_convolutional_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", 'cifar_convolutional_model','Checkpoint dir')

FLAGS = flags.FLAGS



def nn():
    
    return Sequential([Convolution(output_depth=10,input_depth=num_channels,batch_size=FLAGS.batch_size, input_dim=32, act ='relu', stride_size=1, pad='VALID'),
                       AvgPool(),

                       Convolution(output_depth=25,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=4,output_depth=100,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                       AvgPool(),
                       
                       Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID')

                       
    ])


def nn2():
    
    return Sequential([Convolution(output_depth=32,input_depth=num_channels,batch_size=FLAGS.batch_size, input_dim=32, act ='relu', stride_size=1, pad='VALID'),
                       AvgPool(),

                       Convolution(output_depth=25,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),

                       
                       Convolution(kernel_size=4,output_depth=100,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),
                    
                    Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                       AvgPool(),

                       Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID')

                       
    ])

def vgg_like_nn():
    
    return Sequential([Convolution(output_depth=64,input_depth=num_channels,batch_size=FLAGS.batch_size, input_dim=32, act ='relu', stride_size=1, pad='VALID'),
                       Convolution(output_depth=64, act ='relu', stride_size=1, pad='VALID'),
                       MaxPool(),

                        Convolution(output_depth=128,stride_size=1, act ='relu', pad='VALID'),
                        Convolution(output_depth=128,stride_size=1, act ='relu', pad='VALID'),
                        MaxPool(),
                           
                        
                        # Linear(output_dim=1000),
                        # Linear(output_dim=10, input_dim = 1000),
                       
                        Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                        AvgPool(),
                       
                        Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),
                        AvgPool(),
                       
                        Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID')


                       
    ])


def cuda_conv_like_nn():
    
    return Sequential([Convolution(output_depth=64,input_depth=num_channels,batch_size=FLAGS.batch_size, input_dim=32, act ='relu', stride_size=1, pad='VALID'),
                       AvgPool(),
                       Convolution(output_depth=64,stride_size=1, act ='relu', pad='VALID'),
                       AvgPool(),

                        
                    
                       #Convolution(kernel_size=1, output_depth=10,stride_size=1, pad='VALID'),

                        Linear(output_dim=1000),
                        Linear(output_dim=10, input_dim = 1000),
                       
    ])


use_grayscale = True


if(use_grayscale):
    num_channels = 1
    data_function = LoadGSCifarData
else:
    num_channels = 3
    data_function = LoadCifarDataFromImages


model_to_use = nn2
#model_to_use = vgg_like_nn
#model_to_use = cuda_conv_like_nn



def feed_dict(train=False):
    if(train):
        k = 0.9
    else:
        k = 1.0
    xs, ys = data_function(FLAGS.batch_size,train)
    
    
    return xs, ys, k


def train():
  # Import data
  config = tf.ConfigProto(allow_soft_placement = True)
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 32,32,num_channels], name='absolute_input_x')
        y_ = tf.placeholder(tf.float32, [None, 10], name='absolute_output_y')
        keep_prob = tf.placeholder(tf.float32)
            
    with tf.variable_scope('model'):
        net = model_to_use()
       

        inp = tf.pad(tf.reshape(x, [FLAGS.batch_size,32,32,num_channels]), [[0,0],[2,2],[2,2],[0,0]], name='absolute_input')
        op = net.forward(inp)
        y = tf.squeeze(op)
        trainer = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])
    with tf.variable_scope('relevance'):
        if FLAGS.relevance:
            LRP = net.lrp(op, FLAGS.relevance_method, 1e-3)

            # LRP layerwise 
            relevance_layerwise = []
            R = op
            for layer in net.modules[::-1]:
                R = net.lrp_layerwise(layer, R, FLAGS.relevance_method, 1e-3)
                relevance_layerwise.append(R)

        else:
            LRP=[]
            relevance_layerwise = []
            
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

    tf.global_variables_initializer().run()
    
    utils = Utils(sess, FLAGS.checkpoint_reload_dir)
    ''' Reload from a list of numpy arrays '''
    if FLAGS.reload_model:
        reload_using_checkpoint = True

        if(reload_using_checkpoint):
            utils = Utils(sess, FLAGS.checkpoint_reload_dir)
            
            utils.reload_model()
        else:
            tvars = tf.trainable_variables()
            npy_files = np.load('cifar_convolutional_model/model.npy', encoding='bytes')
            [sess.run(tv.assign(npy_files[tt])) for tt,tv in enumerate(tvars)]
    
    for i in range(FLAGS.max_steps):
        if i % FLAGS.test_every == 0:  # test-set accuracy
            d = feed_dict(False)
            test_inp = {x:d[0], y_: d[1], keep_prob: d[2]}
            #pdb.set_trace()
            
            import timeit
            start = timeit.default_timer()

            summary, acc , y1, relevance_test, rel_layer= sess.run([merged, accuracy, y, LRP, relevance_layerwise], feed_dict=test_inp)

            stop = timeit.default_timer()
            print('Runtime: %f' %(stop - start))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %f' % (i, acc))
            #pdb.set_trace()
            print([np.sum(rel) for rel in rel_layer])
            print(np.sum(relevance_test))
            
            # save model if required
            if FLAGS.save_model:
                utils.save_model()

        else:  
            d = feed_dict(True)
            inp = {x:d[0], y_: d[1], keep_prob: d[2]}
            summary, _ , acc, relevance_train,op, rel_layer= sess.run([merged, trainer.train,accuracy, LRP,y, relevance_layerwise], feed_dict=inp)
            print('Accuracy at step %s: %f' % (i, acc))
            train_writer.add_summary(summary, i)
            
            
    # relevances plotted with visually pleasing color schemes
    if FLAGS.relevance:
        #pdb.set_trace()
        relevance_test = relevance_test[:,2:34,2:34,:]
        images = d[0].reshape([FLAGS.batch_size,32,32,num_channels])
        plot_relevances(relevance_test.reshape([FLAGS.batch_size,32,32,num_channels]), images, test_writer )
        # plot train images with relevances overlaid
        # relevance_train = relevance_train[:,2:30,2:30,:]
        # images = inp[inp.keys()[0]].reshape([FLAGS.batch_size,28,28,1])
        # plot_relevances(relevance_train.reshape([FLAGS.batch_size,28,28,1]), images, train_writer )


    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
