from pycocotools import mask as maskUtils
import numpy as np

def encodeMask(mask):
    fortran = np.asfortranarray(mask)
    enc_mask = maskUtils.encode(fortran)

    return enc_mask

def decodeMask(enc_mask):
   return maskUtils.decode(enc_mask).astype(bool)