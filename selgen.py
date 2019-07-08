import pandas as pd
import cv2
import numpy as np
import scipy.ndimage
import scipy.io
import os
import re
import selgen_global

def crop_ROI(image):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

    b,g,r = cv2.split(image)

    M3 = b < 100

    mask = M3

    #sloupce
    proj = np.sum(mask, axis = 1)
    mez = max(proj)/3

    pos = round(len(proj)/2)

    while proj[pos] > mez:
        pos = pos+1

    dos = pos

    pos = round(len(proj)/2)

    while proj[pos] > mez:
        pos = pos-1

    ods = pos

    #radky
    proj = np.sum(mask,axis = 0)
    mez = max(proj)/3
    pos = round(len(proj)/2)

    while proj[pos] >mez:
        pos = pos+1

    dor = pos

    pos = round(len(proj)/2)

    while proj[pos]>mez:
        pos = pos-1

    odr = pos

    ROI = image[ods:dos, odr:dor,:]
    
    return ROI


def half_split(image):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h,w = hsv.shape[0:2]
   
    lower_1 = np.array([0,0,50])
    upper_1 = np.array([50,255,180])

    mask_1 = cv2.inRange(hsv, lower_1, upper_1)
    mask_1 = (mask_1==0).astype('uint8') * 255

    mask_1[0:200,:]=0
    mask_1[h-200:h,:]=0
    mask_1[:,0:1200]=255
    mask_1[:,w-1200:w]=255
    
    mask_1 = (mask_1 > 0).astype('uint8')
    proj_1 = np.sum(mask_1, axis = 0)
    index_1  =  np.argmin(proj_1)
    
    lower_2 = np.array([0,0,20])
    upper_2 = np.array([250,100,150])
    
    mask_2 = cv2.inRange(hsv, lower_2, upper_2)
    mask_2 = (mask_2==0).astype('uint8') * 255


    mask_2[0:200,:]=0
    mask_2[h-200:h,:]=0
    mask_2[:,0:1200]=255
    mask_2[:,w-1200:w]=255

    mask_2 = (mask_2 > 0).astype('uint8')
    proj_2 = np.sum(mask_2, axis = 0)
    index_2  =  np.argmin(proj_2)
    
    mask = mask_1 * mask_2 
    
    proj = np.sum(mask, axis = 0)
    
    index = np.argmin(proj)
    
    if(hsv.sum() < 1000000000 &  0 < proj_1[index_1] < 300):
        
        index = index_1
    
    if(hsv.sum() > 1000000000 &  0 < proj_2[index_2] < 300):
        
        index = index_2   
    
    else:
        
        if(proj_1[index_1] < proj_2[index_2] < 500):
            
            index = index_2
        
        if(500 > proj_1[index_1] > proj_2[index_2]):
            
            index = index_1
    
    index_cor = np.argmin(proj)
    
    index = (index + index_cor) // 2
    
    left = image[:, 0:index, :]
    right = image[:, index:image.shape[1], :]

    return left, right



def find_cross_mask(image, etalon_path):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    assert (type(etalon_path) == str) and (os.path.exists(etalon_path)), 'Path to bucket storage credentials json file is not valid'

    pattern = scipy.io.loadmat(etalon_path)

    cross_pattern = pattern['krizek']
    
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    
    lower = np.array([60,0,50])
    upper = np.array([140,100,155])

    background_mask = cv2.inRange(hsv, lower, upper)
    background_mask = (background_mask>0).astype('uint8')
      
    grayscale_ROI = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, ROI_thresholded = cv2.threshold(grayscale_ROI,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cross_mask = scipy.ndimage.correlate(ROI_thresholded, cross_pattern, mode='nearest')
    
    cross_mask = cross_mask * background_mask
    
    return cross_mask


def get_cross_grid(image, side):
    
    import numpy as np
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 2) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    assert side in ('left','right'), 'Side argument is string left or right'

    h,w = image.shape
    
    image[0:130,:] = 0
    image[h-130:h,:] = 0

    if(side == 'left'):
        
        image[:,0:150] = 0
        image[:,w-100:w] = 0
        
    if(side == 'right'):
        
        image[:,0:100] = 0
        image[:,w-150:w] = 0
    
    rows = np.sum(image, axis = 0)
    cols = np.sum(image, axis = 1)
    row_indexes = []
    col_indexes = []

    rows = list(rows)
    cols = list(cols)

    for i in range(0,7):

        index = np.argmax(rows)
        
        row_indexes.append(index)
        not_indexes = list(np.linspace(index-80,index+80,161))

        for it in not_indexes:
            if(0 <= int(it) <= w-1):
                rows[int(it)] = min(rows)


    for j in range(0,9):

        index = np.argmax(cols)

        col_indexes.append(index)
        not_indexes = list(np.linspace(index-80,index+80,1651))

        for it in not_indexes:
            if(0 <= int(it) <= h-1):
                cols[int(it)] = min(cols)
       
    row_indexes.sort()
    col_indexes.sort()
    
    r_sum = 0
    c_sum = 0
    
    for i in range(1,len(row_indexes)):
        
        r_sum = r_sum + (row_indexes[i] - row_indexes[i-1])   
    
    for j in range(1,len(col_indexes)):    
    
        c_sum = c_sum + (col_indexes[i] - col_indexes[i-1]) 
        
        
    r_shift = r_sum //  (len(row_indexes)-1)   
    c_shift = c_sum //  (len(col_indexes)-1)
    
    min_r = row_indexes[0]-r_shift
    max_r = row_indexes[len(row_indexes)-1]+r_shift
    
    min_c = col_indexes[0]-c_shift
    max_c = col_indexes[len(col_indexes)-1]+c_shift
    
    
    min_r = max(0,min_r)
    max_r = min(max_r,w)
    
    min_c = max(0,min_c)
    max_c = min(max_c,h)
    
    row_indexes.append(min_r)
    row_indexes.append(max_r)
    
    col_indexes.append(min_c)
    col_indexes.append(max_c)

    row_indexes.sort()
    col_indexes.sort() 
    
    coll_indexes = []
    
    for l in range (0,len(col_indexes),2):
        
        coll_indexes.append(col_indexes[l])
        
    return row_indexes,coll_indexes


def split_tray(image,side,row_indexes,col_indexes):
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    assert side in ('left','right'), 'Side argument is string left or right'
    assert type(row_indexes) == list, 'Row_indexes argument is list of integers'
    assert type(col_indexes) == list, 'Cow_indexes argument is list of integers'

    areas = []
    
    class area():
    
        def __init__(self, side, location, cropped_area):
        
            self.side = side
            self.location = location
            self.cropped_area = cropped_area
            
    for i in range(0,len(row_indexes)-1):
        for j in range(0,len(col_indexes)-1):


            cropped_area = image[col_indexes[j]:col_indexes[j+1],row_indexes[i]:row_indexes[i+1],:]
            area_ = area(side,(i,j),cropped_area)
            areas.append(area_)
            
    return areas


def process_selgen_image(image):
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    roi = crop_ROI(image)
    
    left_part, right_part = half_split(roi)
    
    left_part_mask = find_cross_mask(left_part, selgen_global.etalon_path)
    right_part_mask = find_cross_mask(right_part, selgen_global.etalon_path)

    left_part_row, left_part_col = get_cross_grid(left_part_mask, 'left')
    right_part_row, right_part_col = get_cross_grid(right_part_mask, 'right')
   
    left_part_areas = split_tray(left_part,'left', left_part_row, left_part_col)
    right_part_areas = split_tray(right_part,'right', right_part_row, right_part_col)    
    
    areas = [*left_part_areas, *right_part_areas]
    
    return areas


def segmentation_biomass(image, lower_thresh, upper_thresh):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

    h,w = image.shape[0:2]

    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = (mask>0).astype('uint8')


    biomass = mask.sum() / (h*w)

    return biomass


def paint_active_biomass(image, lower_thresh, upper_thresh):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

    
    ROI = crop_ROI(image)

    hsv = cv2.cvtColor(ROI.copy(), cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
    mask = (mask>0).astype('uint8') * 255
    mask = cv2.Canny(mask,100,200)

    ret,cnts,jj = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(ROI, cnts, -1, (0, 0, 255), 1)

    return ROI


def evaluate_selgen_batch(path):

    assert (type(path) == str), 'path should navigate into the folder where batch of images is stored'
    
    if not os.path.exists(path + 'contoured_images/'):
        os.makedirs(path + 'contoured_images/')

    if not os.path.exists(path + 'processed/'):
        os.makedirs(path + 'processed/')

    if not os.path.exists(path + 'unprocessed/'):
        os.makedirs(path + 'unprocessed/')
         
    formats = ('.JPG','.jpg','.PNG','.png','.bmp','.BMP','.TIFF','.tiff','.TIF','.tif')
    files = [file for file in os.listdir(path) if file.endswith(formats)]
    data = []   
        
    for file in files:
        
        #try:
    
        image = cv2.imread(path+file)

        contoured_image = paint_active_biomass(image, selgen_global.lower_thresh, selgen_global.upper_thresh)

        cv2.imwrite(path + 'contoured_images/' + file, contoured_image)

        areas = process_selgen_image(image)

        for area in areas:
            
            biomass = segmentation_biomass(area.cropped_area, selgen_global.lower_thresh, selgen_global.upper_thresh)

            info  = file.split('.')
            regex = re.match('(^\d+)([a-z])', info[0])
            day = regex.group(1)  
            variant = regex.group(2)
            
            side = area.side
            location = area.location

            data.append(dict(zip(('variant','day','side','location','biomass'),(variant, day, side, location, biomass))))
        
        cv2.imwrite(path + 'processed/' + file, image)
        os.remove(path+file)
        print('{} was succesfully processed'.format(file))

        # #except Exception as e:
            
        #     cv2.imwrite(path + 'unprocessed/' + file, image)
        #     os.remove(path+file)
        #     print('{} wasn\'t succesfully processed because of:')
        #     print(e)
                
                
    df = pd.DataFrame(data)
    df.to_excel(path + 'batch_output.xlsx')