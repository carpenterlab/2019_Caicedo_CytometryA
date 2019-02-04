import os
import os.path
import numpy as np

import skimage.io
import keras.preprocessing.image

import utils.augmentation


def setup_working_directories(config_vars):

    ## Expected raw data directories:
    config_vars["raw_images_dir"] = os.path.join(config_vars["root_directory"], 'raw_images/')
    config_vars["raw_annotations_dir"] = os.path.join(config_vars["root_directory"], 'raw_annotations/')

    ## Split files
    config_vars["path_files_training"] = os.path.join(config_vars["root_directory"], 'training.txt')
    config_vars["path_files_validation"] = os.path.join(config_vars["root_directory"], 'validation.txt')
    config_vars["path_files_test"] = os.path.join(config_vars["root_directory"], 'test.txt')

    ## Transformed data directories:
    config_vars["normalized_images_dir"] = os.path.join(config_vars["root_directory"], 'norm_images/')
    config_vars["boundary_labels_dir"] = os.path.join(config_vars["root_directory"], 'boundary_labels/')

    return config_vars

def single_data_from_images(x_dir, y_dir, image_names, batch_size, bit_depth, dim1, dim2, rescale_labels):

    ## Prepare image names
    x_image_names = [os.path.join(x_dir, f) for f in image_names]
    y_image_names = [os.path.join(y_dir, f) for f in image_names]

    ## Load all images in memory
    x = skimage.io.imread_collection(x_image_names).concatenate()
    y = skimage.io.imread_collection(y_image_names).concatenate()

    ## Crop the desired size
    x = x[:, 0:dim1, 0:dim2]
    x = x.reshape(-1, dim1, dim2, 1)
    y = y[:, 0:dim1, 0:dim2, :]

    ## Setup Keras Generators
    rescale_factor = 1./(2**bit_depth - 1)
    
    if(rescale_labels):
        rescale_factor_labels = rescale_factor
    else:
        rescale_factor_labels = 1

    gen_x = keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor)
    gen_y = keras.preprocessing.image.ImageDataGenerator(rescale=rescale_factor_labels)
    
    seed = 42

    stream_x = gen_x.flow(
        x,
        batch_size=batch_size,
        seed=seed
    )
    stream_y = gen_y.flow(
        y,
        batch_size=batch_size,
        seed=seed
    )
    
    flow = zip(stream_x, stream_y)
    
    return flow


def random_sample_generator(x_dir, y_dir, image_names, batch_size, bit_depth, dim1, dim2, rescale_labels):

    do_augmentation = True
    
    # get image names
    print('Training with',len(image_names), 'images.')

    # get dimensions right -- understand data set
    n_images = len(image_names)
    ref_img = skimage.io.imread(os.path.join(y_dir, image_names[0]))

    if(len(ref_img.shape) == 2):
        gray = True
    else:
        gray = False
    
    # rescale images
    rescale_factor = 1./(2**bit_depth - 1)
    if(rescale_labels):
        rescale_factor_labels = rescale_factor
    else:
        rescale_factor_labels = 1
        
    while(True):
        
        if(gray):
            y_channels = 1
        else:
            y_channels = 3
            
        # buffers for a batch of data
        x = np.zeros((batch_size, dim1, dim2, 1))
        y = np.zeros((batch_size, dim1, dim2, y_channels))
        
        # get one image at a time
        for i in range(batch_size):
                       
            # get random image
            img_index = np.random.randint(low=0, high=n_images)
            
            # open images
            x_big = skimage.io.imread(os.path.join(x_dir, image_names[img_index])) * rescale_factor
            y_big = skimage.io.imread(os.path.join(y_dir, image_names[img_index])) * rescale_factor_labels

            # resizing
            #x_big, y_big = utils.augmentation.resize(patch_x, patch_y)


            # get random crop
            start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            patch_x = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] #* rescale_factor
            patch_y = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] #* rescale_factor_labels

            if(do_augmentation):
                
                rand_flip = np.random.randint(low=0, high=2)
                rand_rotate = np.random.randint(low=0, high=4)
                
                # flip
                if(rand_flip == 0):
                    patch_x = np.flip(patch_x, 0)
                    patch_y = np.flip(patch_y, 0)
                
                # rotate
                for rotate_index in range(rand_rotate):
                    patch_x = np.rot90(patch_x)
                    patch_y = np.rot90(patch_y)

                # illumination
                ifactor = 1 + np.random.uniform(-0.75, 0.75)
                patch_x *= ifactor
                    
            # save image to buffer
            x[i, :, :, 0] = patch_x
            
            if(gray):
                y[i, :, :, 0] = patch_y
            else:
                y[i, :, :, 0:y_channels] = patch_y
            
        # return the buffer
        yield(x, y)


