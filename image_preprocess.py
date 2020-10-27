def standardize(x):
    image = tf.truediv(
    tf.subtract(
        x['image'], 
        tf.reduce_min(x['image'])
    ), 
    tf.subtract(
        tf.reduce_max(x['image']), 
        tf.reduce_min(x['image'])
    )
  )
    mask = x['mask']
    return (image,mask)

def reshaping(x):
    """
    Takes a tensor and reshapes it.
    Inputs:
          x (tuple of tensors with shape (240,240))
    Outputs:
          image, mask (tuples of tensors with shape (240,240,1))
    """
    batchsize = 16
    dims = (240,240,1)
    image = tf.reshape(x['image'], [batchsize,dims[0], dims[1], dims[2]])
    mask = tf.reshape(x['mask'], [batchsize,dims[0], dims[1], dims[2]])
    return {'image':image, 'mask': mask}


def randomCrop(x):
    size = [240, 240]
    image = x['image']
    mask = x['mask']
    image = tf.image.random_crop(image, size)
    mask = tf.image.random_crop(mask,size)
    return {'image':image, 'mask': mask}
def voxel_clip(x):
    """
    Clips the image to the 2nd and 98th percentile values.
    inputs:
        img (a numpy array): The image you want to clip
    outputs:
        img (a numpy array): The image with its values clipped
    """
    upper = np.percentile(x['image'], 98)
    lower = np.percentile(x['image'], 2)
    x['image'][x['image'] > upper] = upper
    x['image'][x['image'] < lower] = lower
    return {'image':image, 'mask': mask}
def binarize(x):
    """
    Convert a given mask array from having multiple categories to 1 and 0.
    inputs:
          array (a numpy array): An array containing multiple integer codings for categories. [0,1,2,3]
    outputs:
          array (a numpy array): An array containing only 1s and 0s. 
    """
    mask = tf.where(x['mask']>0, 1, 0)
    image = x['image']
    return {'image':image, 'mask':mask}



def cast(x):
    image = tf.cast(x['image'], tf.float32)
    mask= tf.cast(x['mask'], tf.float32)
    return {'image':image, 'mask': mask}
