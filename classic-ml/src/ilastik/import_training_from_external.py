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


def import_training_from_external(ilastik_project_file, trainmap_dir):
    pf = h5py.File(ilastik_project_file, 'a')

    for num, lane in enumerate(pf.get(INPUT_INFO_PATH).keys()):
        data_id = int(lane[len('lane'):])
        raw_data = pf.get(RAW_DATA_PATH.format(data_id))

        labels = pf.get(LABELS_PATH.format(data_id))

        image_name = os.path.basename(raw_data.get(u'filePath').value.decode())
        training_map = np.expand_dims(sio.imread(os.path.join(trainmap_dir, image_name)), axis=2)

        print(num, lane, training_map.shape)

        if len(training_map.shape) > 2:
            temp_training_map = np.zeros((training_map.shape[0], training_map.shape[1]), dtype=np.uint16)

        ds = labels.create_dataset('block{:04d}'.format(0), data=training_map)
        ds.attrs['blockSlice'] = '[{:d}:{:d},{:d}:{:d},0:1]'.format(0, training_map.shape[0] - 1, 0,
                                                                    training_map.shape[1] - 1)
    pf.close()
