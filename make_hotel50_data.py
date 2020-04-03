import argparse
import os
import statistics

import numpy as np
import pandas as pd
from PIL import Image


def _check_dir(path):
    return os.path.isdir(path)


def get_stats(l):
    length = len(l)
    mean = l.mean()
    median = statistics.median(l)
    max = l.max()
    min = l.min()
    u_v = len(np.unique(l))

    return {'length': length, 'mean': mean, 'median': median, 'max': max, 'min': min, 'unique_values': u_v}


def load_hotels_data(path):
    hotel_label_list = []
    cam_web_list = []
    image_list = []
    super_class_list = []

    org_path = path

    if os.path.exists(os.path.join(path, 'hotel50-image_label.csv')):

        print('Found csv!')

        dataset = pd.read_csv(os.path.join(path, 'hotel50-image_label.csv'))

    else:
        print('File not found, creating csv...')

        path = os.path.join(path, 'images/train/')

        fst_l_d = os.listdir(path)  # e.g. 1 10 11 12

        label = 0
        super_class = 0

        for f_dir in fst_l_d:
            scd_path = os.path.join(path, f_dir)
            print(scd_path)

            if not _check_dir(scd_path):
                continue

            scd_l_d = os.listdir(scd_path)  # e.g. 9645 20303 3291 35913

            for s_dir in scd_l_d:  # All same super_class
                thd_path = os.path.join(scd_path, s_dir)

                if not _check_dir(thd_path):
                    continue

                thd_l_d = os.listdir(thd_path)  # e.g. traffickcam travel_website

                for t_dir in thd_l_d:  # all same labels

                    imagedir_path = os.path.join(thd_path, t_dir)

                    if not _check_dir(imagedir_path):
                        continue

                    if t_dir == 'travel_website':
                        is_website = 1
                    elif t_dir == 'traffickcam':
                        is_website = 0
                    else:
                        print(imagedir_path)
                        raise Exception('FUCK')

                    if not _check_dir(imagedir_path):
                        continue

                    images = os.listdir(imagedir_path)  # e.g. *.jpg

                    for image in images:
                        image_path = os.path.join(imagedir_path, image)
                        image_list.append(image_path[image_path.find('images'):])
                        hotel_label_list.append(label)
                        cam_web_list.append(is_website)
                        super_class_list.append(super_class)

                label += 1
            super_class += 1

        dataset = pd.DataFrame({'image': image_list, 'hotel_label': hotel_label_list, 'super_class': super_class_list,
                                'is_website': cam_web_list})
        dataset.to_csv(os.path.join(org_path, 'hotel50-image_label.csv'), index=False, header=True)

    return dataset


def get_size(df, path):
    sizes = []

    for idx, row in df.iterrows():
        img = Image.open(os.path.join(path, row[0]))
        sizes.append(img.size)

    return sizes


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', help="path")

    args = parser.parse_args()

    df = load_hotels_data(args.path)

    sizes = get_size(df, args.path)

    print('shape 0')
    print(get_stats(np.array(list(zip(*sizes))[0])))
    print('*' * 70)
    print('shape 1')
    print(get_stats(np.array(list(zip(*sizes))[1])))

if __name__ == '__main__':
    main()
