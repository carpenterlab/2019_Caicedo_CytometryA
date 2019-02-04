import h5py
import os
import skimage.io as sio
import numpy as np

"""
    Implements the import feature of ilastik from external file
"""


INPUT_INFO_PATH = 'Input Data/infos/'
RAW_DATA_PATH = 'Input Data/infos/lane{:04d}/Raw Data'
LABELS_PATH = 'PixelClassification/LabelSets/labels{:03d}'


def absolutize_input_path(ilastik_project_file):
    project_file_dir = os.path.dirname(ilastik_project_file)

    pf = h5py.File(ilastik_project_file, 'a')

    for num, lane in enumerate(pf.get('Input Data/infos/').keys()):
        data_id = int(lane[len('labels'):])
        raw_data_file_path = pf.get(u'Input Data/infos/lane{:04d}/Raw Data/filePath/'.format(data_id))
        raw_data_image_path_string = raw_data_file_path.value.decode()

        new_raw_data_image_path = os.path.abspath(os.path.join(project_file_dir, raw_data_image_path_string))
        raw_data_file_path[...] = new_raw_data_image_path

    pf.close
