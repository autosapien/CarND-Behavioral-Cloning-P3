import glob
import time
import os
import numpy as np
import tensorflow as tf
from math import floor
from csv_helpers import load_data_from_csv, save_data_to_csv
from camera_helpers import load_image, save_image, image_rgb_equalize, random_shift_horiz, random_shear
from visualization_helpers import save_image_pair

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('augment_dir', 'augmented_data/', "Directory where the augmented data should be placed")
flags.DEFINE_float('steering_lb', -1, 'Lower bound to search for steering values')
flags.DEFINE_float('steering_ub', -0.5, 'Upper bound to search for steering values')
flags.DEFINE_integer('count', 0, 'Number of samples to isolate, if 0 save identified images 1 to 1')
flags.DEFINE_string('image_label', "label", 'Label for the stored images; used in the file names of the images')
flags.DEFINE_boolean('visualization', True, "If the visualization image of 2 samples that match should be saved")
flags.DEFINE_boolean('stop_at_identification', False, "If true, only shows the num of identified images")

sample_data_dir = 'sample_data/'


def augment_and_store_to_disk(c, l, r, y, num, label='', dir=''):
    augmented_images_rows = []
    count = 0
    while 1:
        for i, img_c in enumerate(c):
            if count >= num:
                break
            unique_id = str(floor(time.time()*10000))
            file_dir = dir+'IMG/'
            filename_c = file_dir + 'aug_c_' + label + '_' + str(count) + '_' + unique_id + '.jpg'
            filename_l = file_dir + 'aug_l_' + label + '_' + str(count) + '_' + unique_id + '.jpg'
            filename_r = file_dir + 'aug_r_' + label + '_' + str(count) + '_' + unique_id + '.jpg'

            # Report where the images are being stored
            if count == 0:
                print("First center image being save to: ", filename_c)

            # apply the same transform to the c,l,r images
            shear = np.random.uniform()
            shift = np.random.uniform()

            if shear > 0.8:
                c[i] = random_shear(c[i], intensity=0.02, fill_mode="constant")
                l[i] = random_shear(l[i], intensity=0.02, fill_mode="constant")
                r[i] = random_shear(r[i], intensity=0.02, fill_mode="constant")
            if shift > 0.2:
                c[i] = random_shift_horiz(c[i], shift_by=0.01, fill_mode="constant")
                l[i] = random_shift_horiz(l[i], shift_by=0.01, fill_mode="constant")
                r[i] = random_shift_horiz(r[i], shift_by=0.01, fill_mode="constant")

            save_image(c[i], filename_c)
            save_image(l[i], filename_l)
            save_image(r[i], filename_r)

            # Report where the images are stored
            if count == 0:
                print("First set of image saved. Continuing...")

            augmented_images_rows.append([filename_c, filename_l, filename_r, y[i]])
            count += 1
        if count >= num:
            break
    if count > 0:  # We have surely worked on some images
        save_data_to_csv(augmented_images_rows, dir + 'augmented_' + unique_id + '.csv')


def build_augmented_dataset(steering_lb, steering_ub, num=500, image_label='', save_dir='',
                            save_visualization=False, stop_at_identification=True):

    # Load all data
    rows = load_data_from_csv(glob.glob(sample_data_dir + '*.csv')[0])
    angles = np.asarray(rows)[:, 3].astype(np.float)

    to_augment = np.logical_and(angles >= steering_lb, angles <= steering_ub)
    nb_to_augment = np.count_nonzero(to_augment)
    print("The num of training samples with steering in [{}, {}]: {}".format(steering_lb, steering_ub, nb_to_augment))

    # Visualize these images
    if save_visualization:
        indexes = np.random.choice(range(len(to_augment)), 2, p=to_augment.astype(np.float)/nb_to_augment)
        save_image_pair((load_image(rows[indexes[0]][0]),
                         "Augment [{},{}] (steer angle {})".format(steering_lb, steering_ub, rows[indexes[0]][3]),
                         None),
                        (load_image(rows[indexes[1]][0]),
                         "Augment [{},{}] (steer angle {})".format(steering_lb, steering_ub, rows[indexes[1]][3]),
                         None),
                        'static/',
                        "augmented_from_{}_to_{}.jpg".format(steering_lb, steering_ub))

    if stop_at_identification:
        exit()

    # Isolation of images that need Augmentation
    rows = np.asarray(rows)
    identified_rows = rows[to_augment]

    # Send for augmentation
    print("Loading data from the identified {} images...".format(nb_to_augment), end="")
    c = []
    l = []
    r = []
    y = []
    for row in identified_rows:
        c.append(load_image(row[0]))
        l.append(load_image(row[1]))
        r.append(load_image(row[2]))
        y.append(float(row[3]))
    c = np.asarray(c)
    l = np.asarray(l)
    r = np.asarray(r)
    y = np.asarray(y)
    print("Done")

    # if count is 0 use the number of identified images
    if num == 0:
        num = nb_to_augment
    print("Saving {} images to augmented {} images...".format(nb_to_augment, num))

    augment_and_store_to_disk(c=c, l=l, r=r, y=y, num=num, label=image_label, dir=save_dir)


def main(_):
    # Create required directories only if we need to save
    if not FLAGS.stop_at_identification:
        if not os.path.isdir(FLAGS.augment_dir):
            print("WARN: Augment directory '{}' not found...".format(FLAGS.augment_dir), end="")
            os.makedirs(FLAGS.augment_dir)
            print("Fixed")

            if not os.path.isdir(FLAGS.augment_dir + 'IMG/'):
                print("WARN: IMG directory not found in '{}'...".format(FLAGS.augment_dir), end="")
                os.makedirs(FLAGS.augment_dir + 'IMG/')
                print("Fixed")

    build_augmented_dataset(steering_lb=FLAGS.steering_lb, steering_ub=FLAGS.steering_ub, num=FLAGS.count,
                            save_dir=FLAGS.augment_dir, image_label=FLAGS.image_label,
                            save_visualization=FLAGS.visualization, stop_at_identification=FLAGS.stop_at_identification)

if __name__ == '__main__':
    tf.app.run()

