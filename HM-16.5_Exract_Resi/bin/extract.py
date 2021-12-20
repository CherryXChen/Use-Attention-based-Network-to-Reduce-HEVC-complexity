import os
import glob
import random
import numpy as np
import data_info as di

# ----------------------------------------------------------------------------------------------------------------------
# To configure: three variables (CONFIG, YUV_PATH_RESI, INFO_PATH)
CONFIG = 'LDP' # coding configuration for HEVC
if CONFIG == 'LDP': # Low-Delay-P
    YUV_PATH_RESI = '/scratch/user/cherryxchen/HEVC/YUV/' # path storing resi_XX.yuv files
    INFO_PATH = '/scratch/user/cherryxchen/HEVC/Inter_Pred/HM-16.5_Resi_Pre/bin/' # path storing Info_XX.dat files
elif CONFIG == 'LDB': # Low-Delay-B
    YUV_PATH_RESI = '/media/F/DataHEVC/LDB_Resi_Pre/'
    INFO_PATH = '/media/F/DataHEVC/LDB_Info/'
elif CONFIG == 'RA': # Random-Access. Here the loaded data are organized in displaying order. However, the generated will be stored in encoding order finally.
    YUV_PATH_RESI = '/media/F/DataHEVC/RA_Resi_Pre/'
    INFO_PATH = '/media/F/DataHEVC/RA_Info/'
# ----------------------------------------------------------------------------------------------------------------------

YUV_NAME_LIST_FULL = di.YUV_NAME_LIST_FULL
YUV_WIDTH_LIST_FULL = di.YUV_WIDTH_LIST_FULL
YUV_HEIGHT_LIST_FULL = di.YUV_HEIGHT_LIST_FULL
YUV_NF = di.YUV_NF_LIST_FULL
YUV_FRAMERATE = di.YUV_FRAMERATE_LIST_FULL

QP_LIST = [22, 27, 32, 37]

INDEX_LIST = [v for v in list(range(12, 123))]

def modi_phw(file, p, w, h, nf, framerate):
    p_new = 'InputFile : ' + YUV_PATH_RESI + p + '\n'
    w_new = 'SourceWidth : ' + str(w) + '\n'
    h_new = 'SourceHeight : ' + str(h) + '\n'
    nf_new = 'FramesToBeEncoded : ' + str(nf) + '\n'
    framerate_new = 'FrameRate : ' + str(framerate) + '\n'
    file_data = ''
    with open(file, 'r') as f:
        for line in f:
            if 'InputFile' in line:
                line = line.replace(line, p_new)
            if 'SourceWidth' in line:
                line = line.replace(line, w_new)
            if 'SourceHeight' in line:
                line = line.replace(line, h_new)
            if 'FramesToBeEncoded' in line:
                line = line.replace(line, nf_new)
            if 'FrameRate' in line:
                line = line.replace(line, framerate_new)
            file_data += line
    
    with open(file, 'w') as f:
        f.write(file_data)
'''
file_path = YUV_PATH_RESI + 'BasketballPass_416x240_50' + '.yuv'
source_path = '/content/YUV_Gene/encoder_yuv_source.cfg'
modi_phw(source_path, file_path, 416, 240, 500)
'''

def modi_qp(file, qp):
    qp_new = 'QP : ' + str(qp) + '\n'
    file_data = ''
    with open(file, 'r') as f:
        lines = f.readlines()
        lines[37] = lines[37].replace(lines[37], qp_new)
    with open(file, 'w') as f:
        f.writelines(lines)
'''
source_path = '/content/YUV_Gene/encoder_lowdelay_P_main.cfg'
modi_qp(source_path, 22)
'''

def modi_name(name_n, qp, nf):
    res_old = INFO_PATH + 'resi.yuv'
    res_new = INFO_PATH + 'resi_' + name_n + '_qp' + str(qp) + '_nf' + str(500) + '.yuv'
    str_old = INFO_PATH + 'str.bin'
    str_new = INFO_PATH + 'str_' + name_n + '.bin'
    os.rename(res_old, res_new)
    os.rename(str_old, str_new)
'''
name = 'BasketballPass_416x240_50'
name_path = '/content/YUV_Gene/BasketballPass_416x240_50'
modi_name(name_path, name)
'''

def run_shell():
    #a = os.system('cd /scratch/user/cherryxchen/HEVC/Inter_Pred/extract_resi2/bin/; chomd 755 TAppEncoderStatic; bash RUN_LDP.sh')
    a = os.system('chomd 755 TAppEncoderStatic; bash RUN_LDP.sh')
    print(a)

if __name__ == '__main__':
    yuv_list = os.listdir(YUV_PATH_RESI)
    
    YUV_NAME_LDP = YUV_NAME_LIST_FULL[INDEX_LIST[0] : INDEX_LIST[-1]+1]
    YUV_H_LDP = YUV_HEIGHT_LIST_FULL[INDEX_LIST[0] : INDEX_LIST[-1]+1]
    YUV_W_LDP = YUV_WIDTH_LIST_FULL[INDEX_LIST[0] : INDEX_LIST[-1]+1]
    YUV_NF_LDP = YUV_NF
    YUV_QP_LDP = QP_LIST
    YUV_FR_LDP = YUV_FRAMERATE

    for i in range(len(yuv_list)):
        print(i)
        print(yuv_list[i])
    #     print(YUV_NAME_LDP[i])
    #     print(YUV_W_LDP[i])
    #     print(YUV_H_LDP[i])
    #     print(YUV_NF_LDP[i])

    for i in range(len(yuv_list)):
        if i > 15:
            name_temp = yuv_list[i].split('.yuv')[0]
            index = YUV_NAME_LDP.index(name_temp)
            h_temp = YUV_H_LDP[index]
            w_temp = YUV_W_LDP[index]
            nf_temp = YUV_NF_LDP[index]
            framerate_temp = YUV_FR_LDP[index]
        
            source_path = INFO_PATH + 'encoder_yuv_source.cfg'
            main_path = INFO_PATH + 'encoder_lowdelay_P_main.cfg'
            
            modi_phw(source_path, yuv_list[i], w_temp, h_temp, nf_temp, framerate_temp)
            print('this is: ', i)
            for j in range(len(YUV_QP_LDP)):
                modi_qp(main_path, YUV_QP_LDP[j])
                run_shell()
                modi_name(name_temp, YUV_QP_LDP[j], nf_temp)