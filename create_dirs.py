import os, shutil
import fileio

def create_dirs(Ntrain, Nval, Ntest, original_dataset_dir, base_dir):

  #original_dataset_dir = 'C:\\svn\\dwatts\\dev\\datasets\\whale_data\\data'
  #original_test_dataset_dir = 'C:\\svn\\dwatts\\dev\\datasets\\whale_data\\data\\test'

  # smaller dataset
  #base_dir = 'C:\\svn\\dwatts\\dev\\dl_with_python\\whale_small\\'

  if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
  os.mkdir(base_dir)

  # define train, validationa and test splits
  train_dir = os.path.join(base_dir, 'train')
  os.mkdir(train_dir)
  validation_dir = os.path.join(base_dir, 'validation')
  os.mkdir(validation_dir)
  test_dir = os.path.join(base_dir, 'test')
  os.mkdir(test_dir)

  # Right and non-right whale training recording
  train_right_dir = os.path.join(train_dir, 'right')
  os.mkdir(train_right_dir)
  train_nonright_dir = os.path.join(train_dir, 'nonright')
  os.mkdir(train_nonright_dir)

  # Right and non-right whale validation recordings
  validation_right_dir = os.path.join(validation_dir, 'right')
  os.mkdir(validation_right_dir)
  validation_nonright_dir = os.path.join(validation_dir, 'nonright')
  os.mkdir(validation_nonright_dir)

  # Right and non-right whale training recordings
  test_right_dir = os.path.join(test_dir, 'right')
  os.mkdir(test_right_dir)
  test_nonright_dir = os.path.join(test_dir, 'nonright')
  os.mkdir(test_nonright_dir)

  # first work out which files are right/non-right and
  # then copy smaller set to the whale_small dir

  # build a train object
  original_train_dir = os.path.join(original_dataset_dir, 'train')
  train = fileio.TrainData(original_dataset_dir+'\\train.csv',base_dir+'train')

  # Copy Ntrain right whale recordings to train_whale_dir
  train_right_fnames = [train.h1[i] for i in range(Ntrain) ]
  for fname in train_right_fnames:
      src = os.path.join(original_train_dir, fname)
      dst = os.path.join(train_right_dir, fname)
      shutil.copyfile(src, dst)

  # Copy Ntrain non-right whale recordings to train_whale_dir
  train_nonright_fnames = [train.h0[i] for i in range(Ntrain) ]
  for fname in train_nonright_fnames:
      src = os.path.join(original_train_dir, fname)
      dst = os.path.join(train_nonright_dir, fname)
      shutil.copyfile(src, dst)

  # Copy next Nval right whale recordings to validation_whale_dir
  val_right_fnames = [train.h1[i] for i in range(Ntrain, Ntrain+Nval) ]
  for fname in val_right_fnames:
      src = os.path.join(original_train_dir, fname)
      dst = os.path.join(validation_right_dir, fname)
      shutil.copyfile(src, dst)

  # Copy next Nval non-right whale recordings to validation_whale_dir
  val_nonright_fnames = [train.h0[i] for i in range(Ntrain, Ntrain+Nval) ]
  for fname in val_nonright_fnames:
      src = os.path.join(original_train_dir, fname)
      dst = os.path.join(validation_nonright_dir, fname)
      shutil.copyfile(src, dst)

  # Copy next Ntest right whale recordings to test_whale_dir
  test_right_fnames = [train.h1[i] for i in range(Ntrain+Nval, Ntrain+Nval+Ntest) ]
  for fname in test_right_fnames:
      src = os.path.join(original_train_dir, fname)
      dst = os.path.join(test_right_dir, fname)
      shutil.copyfile(src, dst)

  # Copy next Ntest non-right whale recordings to test_whale_dir
  test_nonright_fnames = [train.h0[i] for i in range(Ntrain+Nval, Ntrain+Nval+Ntest) ]
  for fname in test_nonright_fnames:
      src = os.path.join(original_train_dir, fname)
      dst = os.path.join(test_nonright_dir, fname)
      shutil.copyfile(src, dst)

  # Create dictionary of training, validation and test ids/labels
  partition = { 'train': train_right_fnames + train_nonright_fnames,
                'validation': val_right_fnames + val_nonright_fnames,
                'test': test_right_fnames + test_nonright_fnames}

  labels_train_right = {key: 1 for key in train_right_fnames}
  labels_train_nonright = {key: 0 for key in train_nonright_fnames}
  labels_val_right = {key: 1 for key in val_right_fnames}
  labels_val_nonright = {key: 0 for key in val_nonright_fnames}
  labels_test_right = {key: 1 for key in test_right_fnames}
  labels_test_nonright = {key: 0 for key in test_nonright_fnames}

  labels = {**labels_train_right,
            **labels_train_nonright,
            **labels_val_right,
            **labels_val_nonright,
            **labels_test_right,
            **labels_test_nonright}

  return partition, labels
