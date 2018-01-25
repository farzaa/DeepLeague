"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default=os.path.join('data', 'data_training_set.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('model_data', 'league_classes.txt'))

YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


debug = False

BATCH_SIZE_1 = 32
BATCH_SIZE_2 = 8
EPOCHS_1 = 5
EPOCHS_2 = 30
EPOCHS_3 = 30

if debug:
    BATCH_SIZE_1 = 2
    BATCH_SIZE_2 = 2
    EPOCHS_1 = 1
    EPOCHS_2 = 1
    EPOCHS_3 = 1

class TrainingData:
    # the dataset is broken up in to "clusters"
    # these are npz files with 20 games worth of data.
    # its not ecnomical to load the entire dataset into one npz files
    # our pc would most likely run of ram to allocate for the massive file.
    #
    # all_npz_files_clusters is a list paths to all the npz clusters.
    # i load these in on a need basis.
    def __init__(self, all_train_npz_clusters, all_val_npz_clusters):

        # set up our clusters
        self.all_train_npz_clusters = all_train_npz_clusters
        self.all_val_npz_clusters = all_val_npz_clusters

        # keep track of which training cluster we have loaded
        self.curr_train_npz_cluster = np.load(all_train_npz_clusters[0])
        self.train_cluster_index = 0

        # keep track of which validation cluster we have loaded
        self.curr_val_npz_cluster = np.load(all_val_npz_clusters[0])
        self.val_cluster_index = 0

        # 90% of images are training, 10% are validation.
        # images and boxes will simply point to the images of the cluster we are currently on
        self.train_images = self.curr_train_npz_cluster['images']
        self.train_boxes = self.curr_train_npz_cluster['boxes']

        # set up validationas images/boxes well.
        self.val_images = self.curr_val_npz_cluster['images']
        self.val_boxes = self.curr_val_npz_cluster['boxes']

        # pointers to handle the images within our batch
        self.train_batch_pointer = 0
        self.val_batch_pointer = 0


    def load_train_cluster(self):

        # to fix #TODO from below
        # left_over_images = []
        # for i in range(self.train_batch_pointer, len(self.train_images)):
        #     left_over_images.append(self.train_images[i])
        # print("Leftover...")


        # first figure out which cluster we're moving to
        # mod length of all_train_npz_clusters keeps us in range
        self.train_cluster_index = (self.train_cluster_index + 1) % len(self.all_train_npz_clusters)
        # then load it
        print("Loading new cluster... ", self.all_train_npz_clusters[self.train_cluster_index])
        self.curr_train_npz_cluster = np.load(self.all_train_npz_clusters[self.train_cluster_index])
        # then append proper images/boxes
        self.train_images = self.curr_train_npz_cluster['images']
        self.train_boxes = self.curr_train_npz_cluster['boxes']
        # finally, reset training pointer
        self.train_batch_pointer = 0

    # do same thing for val as done above for val clusters
    def load_val_cluster(self):
        self.val_cluster_index = (self.val_cluster_index + 1) % len(self.all_val_npz_clusters)
        self.curr_val_npz_cluster = np.load(self.all_val_npz_clusters[self.val_cluster_index])
        self.val_images = self.curr_val_npz_cluster['images']
        self.val_boxes = self.curr_val_npz_cluster['boxes']
        self.val_batch_pointer = 0

    def load_train_batch(self, batch_size):
        while True:
            # print("TBP.. ", self.train_batch_pointer)
            # this means we have reached the end of our cluster and need to load another.
            # TODO: this is sort of bad because we waste the frames left over.
            # ex batch size 32, cluster as 63 images, after loading first 32 images
            # 32 + 32 > 63, so we skip over all this precious data!
            if self.train_batch_pointer + batch_size > len(self.train_images):
                self.load_train_cluster()

            initial_index = self.train_batch_pointer
            end_index = self.train_batch_pointer + batch_size
            images_to_process = self.train_images[initial_index:end_index]
            boxes_to_process = self.train_boxes[initial_index:end_index]
            # print("Boxes to process... ")
            # print(boxes_to_process)
            # processed
            p_images, p_boxes = process_data(images_to_process, boxes_to_process)
            detectors_mask, matching_true_boxes = get_detector_mask(p_boxes, YOLO_ANCHORS)

            self.train_batch_pointer += batch_size
            yield [p_images, p_boxes, detectors_mask, matching_true_boxes],  np.zeros(len(p_images))

    def load_val_batch(self, batch_size):
        while True:
            # fix pointers if they extend to far!
            if self.val_batch_pointer + batch_size > len(self.val_images):
                self.load_val_cluster()

            initial_index = self.val_batch_pointer
            end_index = self.val_batch_pointer + batch_size
            images_to_process = self.val_images[initial_index:end_index]
            boxes_to_process = self.val_boxes[initial_index:end_index]
            # processed
            p_images, p_boxes = process_data(images_to_process, boxes_to_process)
            detectors_mask, matching_true_boxes = get_detector_mask(p_boxes, YOLO_ANCHORS)

            self.val_batch_pointer += batch_size
            yield [p_images, p_boxes, detectors_mask, matching_true_boxes],  np.zeros(len(p_images))

    # total number of batches to run for one epoch
    def get_train_steps(self, batch_size):
        print("Getting train steps...")
        steps = 0
        for cluster in self.all_train_npz_clusters:
            loaded_clust = np.load(cluster)
            steps += len(loaded_clust['images'])
        print(steps / batch_size)
        return int(steps / batch_size)

    # total number of batches to run for validation
    def get_val_steps(self, batch_size):
        print("Getting val steps...")
        steps = 0
        for cluster in self.all_val_npz_clusters:
            loaded_clust = np.load(cluster)
            steps += len(loaded_clust['images'])
        # return int(len(self.val_images) / batch_size)
        print(steps / batch_size)
        return int(steps / batch_size)

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    # custom data saved as a numpy file.
    # data = (np.load(data_path))
    # easy class to handle all the data
    train_clusts = os.listdir('/media/student/DATA/clusters_cleaned/train/')
    val_clusts = os.listdir('/media/student/DATA/clusters_cleaned/val/')

    train_clus_clean  = []
    val_clus_clean = []
    for folder_name in train_clusts:
        train_clus_clean.append('/media/student/DATA/clusters_cleaned/train/' + folder_name)
    for folder_name in val_clusts:
        val_clus_clean.append('/media/student/DATA/clusters_cleaned/val/' + folder_name)

    data = TrainingData(train_clus_clean, val_clus_clean)

    anchors = YOLO_ANCHORS
    model_body, model = create_model(anchors, class_names)

    train(
        model,
        class_names,
        anchors,
        data
    )

    # here i just pass in the val set of images
    images = None
    boxes = None

    images, boxes = process_data(data.val_images[0:500], data.val_boxes[0:500])
    if debug:
        images, boxes = process_data(data.val_images[0:10], data.val_boxes[0:10])
    draw(model_body,
        class_names,
        anchors,
        images,
        image_set='val', # assumes training/validation split is 0.9
        weights_name='trained_stage_3_best.h5',
        save_all=False)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, freeze_body=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def train(model, class_names, anchors, data):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    print("Training on %d images " % (data.get_train_steps(BATCH_SIZE_1) * BATCH_SIZE_1))
    model.fit_generator(data.load_train_batch(BATCH_SIZE_1),
               steps_per_epoch=data.get_train_steps(BATCH_SIZE_1),
               epochs=EPOCHS_1,
               validation_data=data.load_val_batch(BATCH_SIZE_1),
               validation_steps=data.get_val_steps(BATCH_SIZE_1),
               callbacks=[logging])

    model.save_weights('trained_stage_1.h5')
    print("Saved!")

    model_body, model = create_model(anchors, class_names, load_pretrained=False, freeze_body=False)

    model.load_weights('trained_stage_1.h5')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    print("Running second....")
    model.fit_generator(data.load_train_batch(BATCH_SIZE_2),
              steps_per_epoch=data.get_train_steps(BATCH_SIZE_2),
              epochs=EPOCHS_2,
              validation_data=data.load_val_batch(BATCH_SIZE_2),
              validation_steps=data.get_val_steps(BATCH_SIZE_2),
              callbacks=[logging])

    model.save_weights('trained_stage_2.h5')

    # yad2k calls for smaller batches here
    model.fit_generator(data.load_train_batch(BATCH_SIZE_2),
              steps_per_epoch=data.get_train_steps(BATCH_SIZE_2),
              epochs=EPOCHS_3,
              validation_data=data.load_val_batch(BATCH_SIZE_2),
              validation_steps=data.get_val_steps(BATCH_SIZE_2),
              callbacks=[logging, checkpoint, early_stopping])

    model.save_weights('trained_stage_3.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
