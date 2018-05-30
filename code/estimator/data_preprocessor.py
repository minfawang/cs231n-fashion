import io
import os
import sys
import json
import multiprocessing

from PIL import Image, ImageOps
from tqdm import tqdm

TARGET_SIZE=299
  

def is_empty_image(img):
  '''remove empty image, put them to another folder for verification
  I use simple heuristic: if image size is 293*293, then pick it out.
  '''
  if img.size == (293, 293):
    return True
  
  return False


def pad_resize_images(params):
  ''' 
  Load image from train_path. Given label for each of the image.
  
  return:
    mean_all, var_all: for entire training set
    [(mean, var)]: a list of mean var for each labelId
    max_w, max_h, min_w, min_h: image dimensions
  '''
  img_path, out_path, empty_path = params
  
  if not os.path.exists(img_path):
    return
  
  new_size = (TARGET_SIZE, TARGET_SIZE)
  im = Image.open(img_path)
  
  if im is None:
    return
  
  if is_empty_image(im):
    os.rename(img_path, empty_path)
    return
  
  max_len = max(im.size)
  delta_w = max_len-im.size[0]
  delta_h = max_len-im.size[1]
  padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))      
  im = ImageOps.expand(im, padding, fill=(255,255,255))
  im = im.resize(new_size, Image.ANTIALIAS)
  
  im.save(out_path, format='JPEG', quality=90)


if __name__ == '__main__':
  datasets = [
    ('data/train_images/', 'data/train_processed/','data/train_empty/', 1014544),
    ('data/validation_images/', 'data/validation_processed/', 'data/validation_empty/', 9897),
    ('data/test_images/', 'data/test_processed/', 'data/test_empty/', 39706),
  ] 
  
  for dataset, outdir, emptydir, num_exp in datasets:
    print('Processing images from: {}'.format(dataset))
    print('Output processed images to: {}'.format(outdir))
    print('Moving empty images to: {}'.format(emptydir))
    if not os.path.exists(outdir):
      os.makedirs(outdir)
    if not os.path.exists(emptydir):
      os.makedirs(emptydir)
    
    # parse json dataset file
    files = [("%s%d.jpg"%(dataset, i+1), 
              "%s%d.jpg"%(outdir, i+1), 
              "%s%d.jpg"%(emptydir, i+1)) for i in range(num_exp)]

    # download data
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(files)) as progress_bar:
      for _ in pool.imap_unordered(pad_resize_images, files):
        progress_bar.update(1)

