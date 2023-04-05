import os, glob, h5py, pkbar
import numpy as np


def download_modelnet(url, data_root):
    if not os.path.exists(os.path.join(data_root, 'modelnet')):
        # download data
        zipfile = os.path.basename(url)
        os.system('wget %s --no-check-certificate; unzip %s' % (url, zipfile))
        os.system('rm %s' % zipfile)
        # read data
        train_files = glob.glob(os.path.join('modelnet40_ply_hdf5_2048', '*train*.h5'))
        test_files = glob.glob(os.path.join('modelnet40_ply_hdf5_2048', '*test*.h5'))
        train_pcds, train_labels = read_data(train_files)
        test_pcds, test_labels = read_data(test_files)
        os.system('rm -rf %s' % 'modelnet40_ply_hdf5_2048')
        # save data
        save_data(data_root, train_pcds, train_labels, 'train')
        save_data(data_root, test_pcds, test_labels, 'test')
        print('Done!')


def read_data(files):
    all_pcd = []
    all_label = []
    for h5_name in files:
        f = h5py.File(h5_name, 'r+')
        pcd = f['data'][:].astype('float32')
        label = f['label'][:].astype('uint8')
        f.close()
        all_pcd.append(pcd)
        all_label.append(label[:, 0])
    all_pcd = np.concatenate(all_pcd, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_pcd, all_label


def save_data(data_root, pcds, labels, mode):
    pcd_prefix = os.path.join(data_root, 'modelnet', 'pcd', mode)
    label_prefix = os.path.join(data_root, 'modelnet', 'label', mode)
    os.makedirs(pcd_prefix, exist_ok=True)
    os.makedirs(label_prefix, exist_ok=True)
    bar = pkbar.Pbar(name=f'Processing {mode} data...', target=len(pcds))
    for i, (pcd, label) in enumerate(zip(pcds, labels)):
        np.save(os.path.join(pcd_prefix, f'{i:04}'), pcd)  # pcd.shape == (N, 3)
        np.save(os.path.join(label_prefix, f'{i:04}'), label)  # label.shape == ()
        bar.update(i)


if __name__ == '__main__':
    URL = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    DATA_ROOT = './data'
    download_modelnet(URL, DATA_ROOT)
