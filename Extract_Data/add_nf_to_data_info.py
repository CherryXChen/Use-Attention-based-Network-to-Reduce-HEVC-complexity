import numpy as np
import data_info as di
import os
import glob

# ----------------------------------------------------------------------------------------------------------------------
# To configure: three variables (CONFIG, YUV_PATH_RESI, INFO_PATH)
CONFIG = 'LDP' # coding configuration for HEVC
if CONFIG == 'LDP': # Low-Delay-P
    YUV_PATH_RESI = '/content/YUV/' # path storing resi_XX.yuv files
    INFO_PATH = '/content/info/' # path storing Info_XX.dat files
elif CONFIG == 'LDB': # Low-Delay-B
    YUV_PATH_RESI = '/media/F/DataHEVC/LDB_Resi_Pre/'
    INFO_PATH = '/media/F/DataHEVC/LDB_Info/'
elif CONFIG == 'RA': # Random-Access. Here the loaded data are organized in displaying order. However, the generated will be stored in encoding order finally.
    YUV_PATH_RESI = '/media/F/DataHEVC/RA_Resi_Pre/'
    INFO_PATH = '/media/F/DataHEVC/RA_Info/'
# ----------------------------------------------------------------------------------------------------------------------

INDEX_LIST = [v for v in list(range(12, 123))]

def grab_name_list(index_list):
  name_list_full = di.YUV_NAME_LIST_FULL
  name_list = [name_list_full[index] for index in index_list]
  
  return name_list
'''
name_list = grab_name_list(INDEX_LIST)
print(np.shape(name_list))
'''

def grab_nf(name_list, info_path):
  nf = []
  for index in range(len(name_list)):
    file = glob.glob(info_path + 'Info*' + name_list[index] + '*')
    #print(file)
    nf_temp = file[0].split('_Index')
    nf_temp = nf_temp[0].split('_nf')[1]
    nf_temp = int(nf_temp)
    nf.append(nf_temp)
  return nf

name_list = grab_name_list(INDEX_LIST)
nf = grab_nf(name_list, INFO_PATH)
nf = np.array(nf)

YUV_WIDTH_LIST_FULL = di.YUV_WIDTH_LIST_FULL
nf
