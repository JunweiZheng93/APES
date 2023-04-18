import os, glob, h5py, pkbar
import numpy as np


def download_shapenet(url, data_root):
    if not os.path.exists(os.path.join(data_root, 'shapenet')):
        # download data
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('rm %s' % zipfile)
        # read data
        train_files = glob.glob(os.path.join('hdf5_data', '*train*.h5')) + glob.glob(os.path.join('hdf5_data', '*val*.h5'))
        test_files = glob.glob(os.path.join('hdf5_data', '*test*.h5'))
        train_pcds, train_cls_labels, train_seg_labels = read_data(train_files)
        test_pcds, test_cls_labels, test_seg_labels = read_data(test_files)
        os.system('rm -rf %s' % 'hdf5_data')
        # # save data
        save_data(data_root, train_pcds, train_cls_labels, train_seg_labels, 'train')
        save_data(data_root, test_pcds, test_cls_labels, test_seg_labels, 'test')
        print('Done!')


def read_data(files):
    all_pcd = []
    all_cls_label = []
    all_seg_label = []
    for h5_name in files:
        f = h5py.File(h5_name, 'r+')
        pcd = f['data'][:].astype('float32')
        cls_label = f['label'][:].astype('uint8')
        seg_label = f['pid'][:].astype('uint8')
        f.close()
        all_pcd.append(pcd)
        all_cls_label.append(cls_label[:, 0])
        all_seg_label.append(seg_label)
    all_pcd = np.concatenate(all_pcd, axis=0)
    all_cls_label = np.concatenate(all_cls_label, axis=0)
    all_seg_label = np.concatenate(all_seg_label, axis=0)
    return all_pcd, all_cls_label, all_seg_label


def save_data(data_root, pcds, cls_labels, seg_labels, mode):
    pcd_prefix = os.path.join(data_root, 'shapenet', 'pcd', mode)
    cls_label_prefix = os.path.join(data_root, 'shapenet', 'cls_label', mode)
    seg_label_prefix = os.path.join(data_root, 'shapenet', 'seg_label', mode)
    os.makedirs(pcd_prefix, exist_ok=True)
    os.makedirs(cls_label_prefix, exist_ok=True)
    os.makedirs(seg_label_prefix, exist_ok=True)
    bar = pkbar.Pbar(name=f'Processing {mode} data...', target=len(pcds))
    for i, (pcd, cls_label, seg_label) in enumerate(zip(pcds, cls_labels, seg_labels)):
        np.save(os.path.join(pcd_prefix, f'{i:04}'), pcd)  # pcd.shape == (N, 3)
        np.save(os.path.join(cls_label_prefix, f'{i:04}'), cls_label)  # cls_label.shape == ()
        np.save(os.path.join(seg_label_prefix, f'{i:04}'), seg_label)  # label.shape == (N,)
        bar.update(i)


if __name__ == '__main__':
    download_shapenet('https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip', './data')
