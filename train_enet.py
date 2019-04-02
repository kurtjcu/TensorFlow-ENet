import tensorflow as tf
#from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.train import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
#import tensorflow.contrib.opt.python.training.weight_decay_optimizers
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
from get_class_weights import ENet_weighing, median_frequency_balancing
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')   #so matplotlib works without xserver(for server without gui)
import matplotlib.pyplot as plt
slim = tf.contrib.slim

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util.tf_export import tf_export

#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/original', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', True, 'Whether or not to save your images.')
flags.DEFINE_boolean('combine_dataset', False, 'If True, combines the validation with the train dataset.')

#Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 25, 'The batch size used for validation.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')
flags.DEFINE_float('max_learning_rate', 0.1, 'The max learning rate for your training. (cyclical)')
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')
flags.DEFINE_string('optimizer_type', "adam",'use adam or adamw ')
flags.DEFINE_string('learning_rate_type', "exponential",'use exponential or cyclic for learning rate ')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
eval_batch_size = FLAGS.eval_batch_size #Can be larger than train_batch as no need to backpropagate gradients.
combine_dataset = FLAGS.combine_dataset

#Training parameters
initial_learning_rate = FLAGS.initial_learning_rate
num_epochs_before_decay = FLAGS.num_epochs_before_decay
num_epochs =FLAGS.num_epochs
learning_rate_decay_factor = FLAGS.learning_rate_decay_factor
weight_decay = FLAGS.weight_decay
epsilon = 1e-8

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

#Use median frequency balancing or not
weighting = FLAGS.weighting

#Visualization and where to save images
save_images = FLAGS.save_images
photo_dir = os.path.join(FLAGS.logdir, "images")

#Directories
dataset_dir = FLAGS.dataset_dir
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Get the images into a list
image_files = sorted([os.path.join(dataset_dir, 'train', file) for file in os.listdir(dataset_dir + "/train") if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, "trainannot", file) for file in os.listdir(dataset_dir + "/trainannot") if file.endswith('.png')])

image_val_files = sorted([os.path.join(dataset_dir, 'val', file) for file in os.listdir(dataset_dir + "/val") if file.endswith('.png')])
annotation_val_files = sorted([os.path.join(dataset_dir, "valannot", file) for file in os.listdir(dataset_dir + "/valannot") if file.endswith('.png')])

if combine_dataset:
    image_files += image_val_files
    annotation_files += annotation_val_files

#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = len(image_files) / batch_size
num_steps_per_epoch = num_batches_per_epoch
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#=================CLASS WEIGHTS===============================
#Median frequency balancing class_weights
if weighting == "MFB":
    class_weights = median_frequency_balancing()
    print "========= Median Frequency Balancing Class Weights =========\n", class_weights

#Inverse weighing probability class weights
elif weighting == "ENET":
    class_weights = ENet_weighing()
    print "========= ENet Class Weights =========\n", class_weights

#============= TRAINING =================
def weighted_cross_entropy(onehot_labels, logits, class_weights):
    '''
    A quick wrapper to compute weighted cross entropy. 

    ------------------
    Technical Details
    ------------------
    The class_weights list can be multiplied by onehot_labels directly because the last dimension
    of onehot_labels is 12 and class_weights (length 12) can broadcast across that dimension, which is what we want. 
    Then we collapse the last dimension for the class_weights to get a shape of (batch_size, height, width, 1)
    to get a mask with each pixel's value representing the class_weight.

    This mask can then be that can be broadcasted to the intermediate output of logits
    and onehot_labels when calculating the cross entropy loss.
    ------------------

    INPUTS:
    - onehot_labels(Tensor): the one-hot encoded labels of shape (batch_size, height, width, num_classes)
    - logits(Tensor): the logits output from the model that is of shape (batch_size, height, width, num_classes)
    - class_weights(list): A list where each index is the class label and the value of the index is the class weight.

    OUTPUTS:
    - loss(Tensor): a scalar Tensor that is the weighted cross entropy loss output.
    '''
    weights = onehot_labels * class_weights
    weights = tf.reduce_sum(weights, 3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, weights=weights)

    return loss

def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.1,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular',
                         name=None):

    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")

    with ops.name_scope(name, "CyclicLearningRate",
                      [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            cycle = math_ops.floor(math_ops.add(1., global_div_double_step))

            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))

            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)

            if mode == 'triangular2':
                clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
                    cycle-1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)

            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()

        return cyclic_lr

def run():
    
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TRAINING BRANCH=======================
        #Load the files into one input queue
        
        images = tf.convert_to_tensor(image_files)
        print'1'
        annotations = tf.convert_to_tensor(annotation_files)
        print'2'
        input_queue = tf.train.slice_input_producer([images, annotations]) #Slice_input producer shuffles the data by default.
        print'3'

        #Decode the image and annotation raw content
        image = tf.read_file(input_queue[0])
        print'4'
        image = tf.image.decode_image(image, channels=3)
        print'5'
        annotation = tf.read_file(input_queue[1])
        print'6'
        annotation = tf.image.decode_image(annotation)
        print'7'

        #preprocess and batch up the image and annotation
        preprocessed_image, preprocessed_annotation = preprocess(image, annotation, image_height, image_width)
        print'8'
        images, annotations = tf.train.batch([preprocessed_image, preprocessed_annotation], batch_size=batch_size, allow_smaller_final_batch=True)
        print'9'

        #Create the model inference
        with slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):
            logits, probabilities = ENet(images,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         stage_two_repeat=stage_two_repeat,
                                         skip_connections=skip_connections)

        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations = tf.reshape(annotations, shape=[batch_size, image_height, image_width])
        print'10'
        annotations_ohe = tf.one_hot(annotations, num_classes, axis=-1)
        print'11'

        #Actually compute the loss
        loss = weighted_cross_entropy(logits=logits, onehot_labels=annotations_ohe, class_weights=class_weights)
        print'12'
        total_loss = tf.losses.get_total_loss()
        print'13'

        #Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()
        print'14'

        #Define your exponentially decaying learning rate

        #############################################################
        #############################################################
        #############################################################
        #              LEarning rate...

        if(FLAGS.learning_rate_type == "exponential"):
            print'learning rate is exp'
            lr = tf.train.exponential_decay(
                learning_rate = initial_learning_rate,
                global_step = global_step,
                decay_steps = decay_steps,
                decay_rate = learning_rate_decay_factor,
                staircase = True)
        elif(FLAGS.learning_rate_type == "cyclic"):
            print'learning rate is cyclic'
            lr = cyclic_learning_rate(
                learning_rate = 6.2e-4, #initial_learning_rate,  #0.01
                global_step = global_step,
                max_lr= 2.5e-3, #max_learning_rate , #0.1
                step_size= num_steps_per_epoch,  #2-8 x training iterations in epoch.
                gamma=0.99994,
                mode='triangular',
                name=None)


        print'15'

    
        #Now we can define the optimizer that takes on the learning rate
        #optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        if(FLAGS.optimizer_type == "adam"):
            print'using adam'
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        elif(FLAGS.optimizer_type == "adamw"):
            print'using adamw'
            # myOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer, weight_decay=weight_decay)
            # optimizer = myOptimizer
            MyAdamW = tf.contrib.opt.extend_with_decoupled_weight_decay(tf.train.AdamOptimizer)
            # Create a MyAdamW object
            optimizer = MyAdamW(weight_decay=weight_decay, learning_rate=lr)    #weight_decay=0.001, learning_rate=0.001


        print'16'

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        print'17'

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)
        precision, precision_update = tf.metrics.precision( annotations, predictions)
        accuracy, accuracy_update = tf.metrics.accuracy( annotations, predictions)
        mean_IOU, mean_IOU_update = tf.metrics.mean_iou(predictions=predictions, labels=annotations, num_classes=num_classes)

        metrics_op = tf.group(accuracy_update, mean_IOU_update, precision_update)
        print'21'

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, metrics_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run
            start_time = time.time()
            total_loss, global_step_count, accuracy_val, mean_IOU_val, precision_val, _ = sess.run([train_op, global_step, accuracy, mean_IOU, precision, metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to show some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)    Current Streaming Accuracy: %.4f    Current Mean IOU: %.4f    Current Precision: %.4f', global_step_count, total_loss, time_elapsed, accuracy_val, mean_IOU_val, precision_val)

            return total_loss, accuracy_val, mean_IOU_val, precision_val

        #================VALIDATION BRANCH========================
        #Load the files into one input queue
        images_val = tf.convert_to_tensor(image_val_files)
        print'22'
        annotations_val = tf.convert_to_tensor(annotation_val_files)
        print'23'
        input_queue_val = tf.train.slice_input_producer([images_val, annotations_val])
        print'24'

        #Decode the image and annotation raw content
        image_val = tf.read_file(input_queue_val[0])
        print'25'
        image_val = tf.image.decode_jpeg(image_val, channels=3)
        print'26'
        annotation_val = tf.read_file(input_queue_val[1])
        print'27'
        annotation_val = tf.image.decode_png(annotation_val)
        print'28'

        #preprocess and batch up the image and annotation
        preprocessed_image_val, preprocessed_annotation_val = preprocess(image_val, annotation_val, image_height, image_width)
        print'29'
        images_val, annotations_val = tf.train.batch([preprocessed_image_val, preprocessed_annotation_val], batch_size=eval_batch_size, allow_smaller_final_batch=True)
        print'30'

        with slim.arg_scope(ENet_arg_scope(weight_decay=weight_decay)):
            logits_val, probabilities_val = ENet(images_val,
                                                 num_classes,
                                                 batch_size=eval_batch_size,
                                                 is_training=True,
                                                 reuse=True,
                                                 num_initial_blocks=num_initial_blocks,
                                                 stage_two_repeat=stage_two_repeat,
                                                 skip_connections=skip_connections)
        print'31'
        #perform one-hot-encoding on the ground truth annotation to get same shape as the logits
        annotations_val = tf.reshape(annotations_val, shape=[eval_batch_size, image_height, image_width])
        print'32'
        annotations_ohe_val = tf.one_hot(annotations_val, num_classes, axis=-1)
        print'33'

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded. ----> Should we use OHE instead?
        predictions_val = tf.argmax(probabilities_val, -1)
        precision_val, precision_val_update = tf.metrics.precision(annotations_val, predictions_val)
        accuracy_val, accuracy_val_update = tf.metrics.accuracy( annotations_val, predictions_val)
        mean_IOU_val, mean_IOU_val_update = tf.metrics.mean_iou(predictions=predictions_val, labels=annotations_val, num_classes=num_classes)
        metrics_op_val = tf.group(accuracy_val_update, mean_IOU_val_update, precision_val_update)

        #Create an output for showing the segmentation output of validation images
        segmentation_output_val = tf.cast(predictions_val, dtype=tf.float32)
        segmentation_output_val = tf.reshape(segmentation_output_val, shape=[-1, image_height, image_width, 1])
        segmentation_ground_truth_val = tf.cast(annotations_val, dtype=tf.float32)
        segmentation_ground_truth_val = tf.reshape(segmentation_ground_truth_val, shape=[-1, image_height, image_width, 1])

        def eval_step(sess, metrics_op):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, accuracy_value, mean_IOU_value, precision_value = sess.run([metrics_op, accuracy_val, mean_IOU_val, precision_val])
            time_elapsed = time.time() - start_time

            #Log some information
            logging.info('---VALIDATION--- Validation Accuracy: %.4f    Validation Mean IOU: %.4f    precision: (%.4f)   (%.2f sec/step)', accuracy_value, mean_IOU_value, precision_value, time_elapsed)

            return accuracy_value, mean_IOU_value, precision_value

        #=====================================================

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('Monitor/Total_Loss', total_loss)
        tf.summary.scalar('Monitor/validation_precision', precision_val)
        tf.summary.scalar('Monitor/training_precision', precision)
        tf.summary.scalar('Monitor/validation_accuracy', accuracy_val)
        tf.summary.scalar('Monitor/training_accuracy', accuracy)
        tf.summary.scalar('Monitor/validation_mean_IOU', mean_IOU_val)
        tf.summary.scalar('Monitor/training_mean_IOU', mean_IOU)
        tf.summary.scalar('Monitor/learning_rate', lr)
        tf.summary.image('Images/Validation_original_image', images_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_output', segmentation_output_val, max_outputs=1)
        tf.summary.image('Images/Validation_segmentation_ground_truth', segmentation_ground_truth_val, max_outputs=1)
        my_summary_op = tf.summary.merge_all()
        print'51'

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=logdir, summary_op=None, init_fn=None)
        print'created supervisor'

        # Run the managed session
        with sv.managed_session() as sess:
            for step in xrange(int(num_steps_per_epoch * num_epochs)):
                #At the start of every epoch, show the vital information:
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value = sess.run([lr])
                    logging.info('Current Learning Rate: %s', learning_rate_value)

                #Log the summaries every 10 steps or every end of epoch, which ever lower.
                if step % min(num_steps_per_epoch, 10) == 0:
                    loss, training_accuracy, training_mean_IOU, training_precision = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

                    #Check the validation data only at every third of an epoch
                    if step % (num_steps_per_epoch / 3) == 0:
                        for i in xrange(len(image_val_files) / eval_batch_size):
                            validation_accuracy, validation_mean_IOU, validation_precision = eval_step(sess, metrics_op_val)

                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)
                    
                #If not, simply run the training step
                else:
                    loss, training_accuracy, training_mean_IOU, training_precision = train_step(sess, train_op, sv.global_step, metrics_op=metrics_op)

            #We log the final training loss
            logging.info('Final Loss: %s', loss)
            logging.info('Final Training Precision: %s', training_precision)
            logging.info('Final Training Accuracy: %s', training_accuracy)
            logging.info('Final Training Mean IOU: %s', training_mean_IOU)
            logging.info('Final Validation Precision: %s', validation_precision)
            logging.info('Final Validation Accuracy: %s', validation_accuracy)
            logging.info('Final Validation Mean IOU: %s', validation_mean_IOU)

            #Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)

                #Plot the predictions - check validation images only
                logging.info('Saving the images now...')
                predictions_value, annotations_value = sess.run([predictions_val, annotations_val])

                for i in xrange(eval_batch_size):
                    predicted_annotation = predictions_value[i]
                    annotation = annotations_value[i]

                    plt.subplot(1,2,1)
                    plt.imshow(predicted_annotation)
                    plt.subplot(1,2,2)
                    plt.imshow(annotation)
                    plt.savefig(photo_dir+"/image_" + str(i))

if __name__ == '__main__':
    run()