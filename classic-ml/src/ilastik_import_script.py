import ilastik.import_training_from_external as itfe
import ilastik.absolutize_input_path as iaip
import argparse
import sys
from shutil import copy
import os.path

'''
    Import external training set for each image of the input data to a copy of existing ilastik project file.
    The script expects 2 arguments:
        ilastik_project_file: existing ilastik project file with selected input data, features and labels;
        train_data_dir: directory that contains grayscale label images, where each label corresponds to 
            a label from the ilastik_project_file.
'''

parser = argparse.ArgumentParser()
parser.add_argument('ilastik_project_file', help="ilastik project file path")
parser.add_argument('train_data_dir', help="directory path of training label images")
args = parser.parse_args()

old_project_file = args.ilastik_project_file
project_dir = os.path.dirname(old_project_file)
old_project_file_name = os.path.basename(old_project_file)

train_data_dir = args.train_data_dir
# print(train_data_dir)

if train_data_dir.endswith('/'):
    # print(os.path.split(train_data_dir[:-1]))
    test_name = os.path.split(train_data_dir[:-1])[-1]
    pass
else:
    # print(os.path.split(train_data_dir))
    test_name = os.path.split(train_data_dir)[-1]
    pass

print(test_name)

new_project_file = os.path.join(project_dir, old_project_file_name.split('.')[0]+ '_' + test_name + '.' + old_project_file_name.split('.')[-1])

print('{} -> {}'.format(old_project_file, new_project_file))

copy(old_project_file, new_project_file)

iaip.absolutize_input_path(new_project_file)

itfe.import_training_from_external(new_project_file, args.train_data_dir)
