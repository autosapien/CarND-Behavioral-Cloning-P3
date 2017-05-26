import glob
import os
import random
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
from model import simplified_nvidia_model_with_dropout, nvidia_model_with_dropout
from visualization_helpers import save_image_pair, save_steering_histogram, save_steering_signal, save_training_images
from csv_helpers import load_data_from_csv, save_data_to_csv
from process_camera_image import process_camera_image
from camera_helpers import load_image, show_image_color, image_shift_horiz
from camera_helpers import save_image, random_brightness, image_shear

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'sample_data/', "Directory where the sample data is")
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train')
flags.DEFINE_integer('batch', 32, 'Size of each training batch')
flags.DEFINE_string('model_out', 'model.h5', 'File to save the model')
flags.DEFINE_boolean('resume', False, 'If Training should resume')
flags.DEFINE_string('model_in', 'model.h5', 'The model to use when resuming')
flags.DEFINE_string('session', '0', 'The session name, else uses timestamp in sec')


def load_data_for_training(data_dir, validation_split=0.2, session='abc'):

    # Load data
    rows = load_rows_from_dir(data_dir)

    # Visualize the training data distribution
    title = 'Distribution of Steering Angles in Training Data'
    save_steering_histogram([r[3] for r in rows], title, 'static/histogram_steering_' + session + '.jpg')
    save_steering_signal([r[3] for r in rows], (0, 1000), 'static/signal_steering_0_' + session + '.jpg')
    save_steering_signal([r[3] for r in rows], (3200, 4200), 'static/signal_steering_1_' + session + '.jpg')

    # Split for validation
    train_rows, validation_rows = train_test_split(rows, test_size=validation_split)

    # process three images to see its shape, visualization
    for i in range(3):
        index = random.randint(0, len(train_rows) - 1)
        source_image = load_image(train_rows[index][np.random.randint(0, 3)])
        processed_image = process_camera_image(source_image)
        save_image_pair((source_image, "Source Image", None),
                        (processed_image, "Processed Image", None),
                        'static/',
                        'processed_' + session + '_' + str(i) + '.jpg')

    return train_rows, validation_rows, processed_image.shape


def load_rows_from_dir(data_dir):
    # Load the data from both the sample and collected data
    csv_files = glob.glob(data_dir + '*.csv')
    rows = []
    for csv_file in csv_files:
        rows.extend(load_data_from_csv(csv_file))
    return rows


def train_model(model, train_rows, validation_rows, batch_size=32, epochs=5, save_checkpoints=False):
    train_generator = generator_train(train_rows, batch_size=batch_size)
    validation_generator = generator_validation(validation_rows, batch_size=batch_size)
    checkpoints = []
    if save_checkpoints:
        checkpoints.append(ModelCheckpoint("models/model_new-{epoch:02d}.h5"))

    model.fit_generator(train_generator,
                        samples_per_epoch=10240,
                        validation_data=validation_generator,
                        nb_val_samples=2048,
                        nb_epoch=epochs,
                        callbacks=checkpoints)


def generator_train(datarows, batch_size=32):
    while 1:  # Loop forever so the generator never terminates
        x_train = []
        y_train = []
        for i in range(batch_size):
            row = datarows[np.random.randint(0, len(datarows))]
            steer_angle = float(row[3])

            # Pick images with greater steer angles to handle straight driving bias

            # Pick an image left, right or center, 3x for center cam
            i = np.random.randint(1, 6)
            if i <= 3:
                image = load_image(row[0])
            if i == 4:
                image = load_image(row[1])
                steer_angle += 0.15
            if i == 5:
                image = load_image(row[2])
                steer_angle -= 0.15

            # Flip image horizontally to balance sample data driving bias 50%
            if np.random.uniform() > 0.5:
                image = np.fliplr(image)
                steer_angle *= -1

            # Alter brightness 40%
            #if np.random.uniform() > 0.6:
            #    image = random_brightness(image, (0.8, 1.4))

            # Shift horizontally 50%
            #if np.random.uniform() > 0.5:
            #    shift_by = np.random.uniform(-0.06, 0.06)  # max 25/360 as we are chopping 25 on each side
            #    image = image_shift_horiz(image, shift_by, fill_mode='constant')
            #    steer_angle += (-0.5)*shift_by

            # Shearing 20%
            #if np.random.uniform() > 0.8:
            #    shear_by = np.random.uniform(-0.1, 0.1)
            #    image = image_shear(image, shear_by, fill_mode='constant')
            #    steer_angle += -0.5*shear_by

            x_train.append(process_camera_image(image))
            y_train.append(steer_angle)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        yield x_train, y_train


def generator_validation(datarows, batch_size=32):
    while 1:  # Loop forever so the generator never terminates
        x_train = []
        y_train = []
        for i in range(batch_size):
            row = datarows[np.random.randint(0, len(datarows))]
            x_train.append(process_camera_image(load_image(row[0])))
            y_train.append(float(row[3]))
        yield np.asarray(x_train), np.asarray(y_train)


def main(_):
    # Load data
    train_rows, validation_rows, input_shape = load_data_for_training(data_dir=FLAGS.data_dir,
                                                                      session=FLAGS.session)
    print("Model input shape is: ", input_shape)
    print("Train Samples: : ", len(train_rows))
    print("Validation Samples: ", len(validation_rows))

    # visualization
    i = 0
    imgs = []
    steer = []
    for res in generator_train(train_rows, batch_size=1):
        if i >= 16:
            break
        imgs.append(((res[0][0]+0.5)*255).astype(np.uint8))
        steer.append(res[1][0])
        i += 1
    save_training_images(imgs, steer, 'static/training_' + FLAGS.session + '.jpg')

    # Build / Load model
    if FLAGS.resume:
        if not os.path.isfile(FLAGS.model_in):
            print("ERROR: Input model file '{}' does not exist", FLAGS.model_in)
            exit()
        print("Resuming training from model {}...".format(FLAGS.model_in))
        print()
        model = load_model(FLAGS.model_in)
        plot(model, to_file='static/model.png', show_shapes=True)
    else:
        print("Starting training into model {}".format(FLAGS.model_in))
        print()
        model = simplified_nvidia_model_with_dropout(input_shape)
        plot(model, to_file='static/model.png', show_shapes=True)

    # Train and save
    model.summary()
    train_model(model=model, train_rows=train_rows, validation_rows=validation_rows,
                batch_size=FLAGS.batch, epochs=FLAGS.epochs, save_checkpoints=True)
    model.save(FLAGS.model_out)


if __name__ == '__main__':
    tf.app.run()


