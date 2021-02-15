import pandas as pd
import cv2
import numpy as np
import scipy.ndimage
import scipy.io
import os
import re
import selgen_global

def crop_ROI(image):

    #crop_ROI function localize region of interest = black tray with crop .
    #Localization is based on gradient between white background and black borders of tray 
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    try:

        b,g,r = cv2.split(image)

        M3 = b < 100

        mask = M3

        #columns
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

        #rows
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

    except Exception as e:

        raise e


def half_split(image):

    #half_split function separate left and right part of tray.
    #This step of analysis can be potentionally problematic, with changing light conditions.

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
  
    try:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h,w = hsv.shape[0:2]

        lower = np.array([0,0,20])
        upper = np.array([250,100,150])

        mask = cv2.inRange(hsv, lower, upper)
        mask = (mask==0).astype('uint8')

        mask[0:200,:]=0
        mask[h-200:h,:]=0
        mask[:,0:1200]=1
        mask[:,w-1200:w]=1

        mask = (mask > 0).astype('uint8')
        proj = np.sum(mask, axis = 0)
        index  =  np.round(np.mean(np.argpartition(proj, 30)[:30]),-1).astype('uint16')
        
        left = image[:, 0:index, :]
        right = image[:, index:image.shape[1], :]

        return left, right, index

    except Exception as e: 
        
        raise e

def find_grid_mask(image, etalon_path, side):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    assert (type(etalon_path) == str) and (os.path.exists(etalon_path)), 'Path to bucket storage credentials json file is not valid'

    #find_grid_mask function identifies grid on given part of tray
    #computation of grid with scipy.ndimage.correlate is most demanding part of image analysis

    try:

        pattern = scipy.io.loadmat(etalon_path)

        cross_pattern = pattern['krizek']

        image[0:50,:,:] = 0
        image[image.shape[0]-50:image.shape[0],:,:] = 0
        
        if(side == 'right'):
            
            image[:,image.shape[1]-50:image.shape[1],:] = 0
            
        if(side == 'left'):
            
            image[:,0:50,:] = 0
        
        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        
        lower = np.array([60,0,50])
        upper = np.array([140,100,155])

        background_mask = cv2.inRange(hsv, lower, upper)
        background_mask = (background_mask>0).astype('uint8')
        
        grayscale_ROI = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, ROI_thresholded = cv2.threshold(grayscale_ROI,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        grid_mask = scipy.ndimage.correlate(ROI_thresholded, cross_pattern, mode='nearest')

        #if(side == 'right'):
            
            #cross_mask[:,50:130] = 0
            
        #if(side == 'left'):
            
            #cross_mask[:,cross_mask.shape[1]-130:cross_mask.shape[1]-50] = 0
        
        return grid_mask * background_mask

    except Exception as e:

        raise e

def get_grid_coords(grid_mask, side):

    #get_grid_coords identify coordinates of grid on given part of tray with grid mask
        
    assert (type(grid_mask) == np.ndarray) & (len(grid_mask.shape) == 2) & np.amin(grid_mask) >= 0 & np.amax(grid_mask) <= 255, 'Input data has to be RGB image'
    assert side in ('left','right'), 'Side argument is string left or right'

    try:

        h,w = grid_mask.shape

        grid_mask[0:130,:] = 0
        grid_mask[h-130:h,:] = 0

        if(side == 'left'):

            grid_mask[:,0:150] = 0
            grid_mask[:,w-100:w] = 0

        if(side == 'right'):

            grid_mask[:,0:100] = 0
            grid_mask[:,w-150:w] = 0

        rows = np.sum(grid_mask, axis = 1)
        cols = np.sum(grid_mask, axis = 0)

        row_indexes = []
        col_indexes = []

        rows = list(rows)
        cols = list(cols)

        for i in range(0,7):

            index = np.argmax(cols)

            col_indexes.append(index)
            not_indexes = list(np.linspace(index-80,index+80,161))

            for it in not_indexes:
                if(0 <= int(it) <= w-1):
                    cols[int(it)] = min(cols)


        for j in range(0,9):

            index = np.argmax(rows)

            row_indexes.append(index)
            not_indexes = list(np.linspace(index-80,index+80,161))

            for it in not_indexes:
                if(0 <= int(it) <= h-1):
                    rows[int(it)] = min(rows)

        row_indexes.sort()
        col_indexes.sort()

        r_sum = 0
        c_sum = 0

        for i in range(1,len(row_indexes)):

            r_sum = r_sum + (row_indexes[i] - row_indexes[i-1])   

        for j in range(1,len(col_indexes)):    

            c_sum = c_sum + (col_indexes[j] - col_indexes[j-1]) 


        r_shift = r_sum //  (len(row_indexes)-1)   
        c_shift = c_sum //  (len(col_indexes)-1)

        min_r = row_indexes[0]-r_shift
        max_r = row_indexes[len(row_indexes)-1]+r_shift

        min_c = col_indexes[0]-c_shift
        max_c = col_indexes[len(col_indexes)-1]+c_shift


        min_r = max(0,min_r)
        max_r = min(max_r,h)

        min_c = max(0,min_c)
        max_c = min(max_c,w)

        row_indexes.append(min_r)
        row_indexes.append(max_r)

        col_indexes.append(min_c)
        col_indexes.append(max_c)


        row_indexes.sort()
        col_indexes.sort() 

        roww_indexes = []

        for l in range (0,len(row_indexes),2):

            roww_indexes.append(row_indexes[l])
        
        return roww_indexes,col_indexes

    except Exception as e:

        raise e


def split_cells(image, side, row_indexes, col_indexes):
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    assert side in ('left','right'), 'Side argument is string left or right'
    assert type(row_indexes) == list, 'Row_indexes argument is list of integers'
    assert type(col_indexes) == list, 'Cow_indexes argument is list of integers'

    #split_cells function separate part of tray with identified coordinates in cells of crop growth

    try:

        areas = []
        
        class area():
        
            def __init__(self, side, row, column, cropped_area, size):
            
                self.side = side
                self.row = row
                self.column = column
                self.cropped_area = cropped_area
                self.size = size
                
        for i in range(0,len(row_indexes)-1):
            for j in range(0,len(col_indexes)-1):


                cropped_area = image[row_indexes[i]:row_indexes[i+1],col_indexes[j]:col_indexes[j+1],:]
                area_ = area(side,i,j,cropped_area,cropped_area.shape[0:2])
                areas.append(area_)
                
        return areas

    except Exception as e:

        raise e


def process_image(image):
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    #process_image function performs all previous functions/steps and returns tray area and list of crop growth cells for left and right part
    #this function also paints grid of tray with red lines for image visual result

    try:

        roi = crop_ROI(image)
        
        left_part, right_part, index = half_split(roi)
        
        left_part_mask = find_grid_mask(left_part, selgen_global.etalon_path, 'left')
        right_part_mask = find_grid_mask(right_part, selgen_global.etalon_path, 'right')

        left_part_row, left_part_col = get_grid_coords(left_part_mask, 'left')
        right_part_row, right_part_col = get_grid_coords(right_part_mask, 'right')
    
        left_part_areas = split_cells(left_part,'left', left_part_row, left_part_col)
        right_part_areas = split_cells(right_part,'right', right_part_row, right_part_col)    

        for line in left_part_row:

            cv2.line(roi, (left_part_col[0], line), (left_part_col[-1], line), (255,0,0), 2)

        for line in left_part_col:

            cv2.line(roi, (line, left_part_row[0]), (line, left_part_row[-1]), (255,0,0), 2)

        for line in right_part_row:

            cv2.line(roi, (right_part_col[0]+index, line), (right_part_col[-1]+index, line), (255,0,0), 2)

        for line in right_part_col:

            line = line + index
            cv2.line(roi, (line, right_part_row[0]), (line, right_part_row[-1]), (255,0,0), 2)
        

        areas = left_part_areas + right_part_areas
        
        return areas, roi

    except Exception as e:

        raise e


def segmentation_biomass(image, lower_thresh, upper_thresh):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

    #segmentation_biomass function segment crop in given area with predefined thresholds in HSV color space

    try:

        h,w = image.shape[0:2]

        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
        mask = (mask>0).astype('uint8')


        biomass = mask.sum() / (h*w)

        return biomass

    except Exception as e:
        
        raise e


def paint_active_biomass(image, lower_thresh, upper_thresh):

    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'

    #paint_active_biomass function draw contours around active crop biomass in a whole image

    try:

        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_thresh, upper_thresh)
        mask = (mask>0).astype('uint8') * 255
        mask = cv2.Canny(mask,100,200)

        cnts, __ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, cnts, -1, (0, 0, 255), 1)

        return image

    except Exception as e:

        raise e 


def filename_parser(file):

    assert type(file) == str, 'Argument of function has to be string'

    #filename_parser function parse important data from image filename

    try:

        info  = file.split('.')[0].split('_')

        return {'variant': info[0], 'date': info[2], 'time': info[4]}

    except Exception as e:

        raise

if __name__ == '__main__':

    assert os.path.exists(selgen_global.path) , 'path should navigate into the folder where batch of images are stored'
    
    if not os.path.exists(selgen_global.path + 'results/'):
        os.makedirs(selgen_global.path + 'results/')
 
         
    formats = ('.JPG','.jpg','.PNG','.png','.bmp','.BMP','.TIFF','.tiff','.TIF','.tif')
    files = [file for file in os.listdir(selgen_global.path) if file.endswith(formats)]
    data = [] 

    print('#########################################')
    print('ANALYSIS OF {} IMAGE BATCH STARTED'.format(len(files)))
    print('#########################################')

    f = open(selgen_global.path + 'results/' + "failures.txt","w+")  
        
    for i, file in enumerate(files):

        print('{} of {} images was processed'.format(i,len(files)))
        
        try:
            
            metadata = filename_parser(file)
            
            image = cv2.imread(selgen_global.path+file)

            areas, roi = process_image(image)

            contoured_image = paint_active_biomass(roi, selgen_global.lower_thresh, selgen_global.upper_thresh)

            cv2.imwrite(selgen_global.path + 'results/' + file, contoured_image)

            for area in areas:
                
                biomass = segmentation_biomass(area.cropped_area, selgen_global.lower_thresh, selgen_global.upper_thresh)

                data.append(dict(zip(('date','time','variant','side','row', 'column','biomass', 'size'),(metadata['date'], metadata['time'], metadata['variant'], area.side, area.row, area.column, biomass, area.size))))
            
            print('{} was succesfully processed.'.format(file))

        except Exception as e:

            print('{} processing failed'.format(file))
            f.write(file + ': \t' + str(e) + '\n')
                
    df = pd.DataFrame(data)
    df.to_excel(selgen_global.path + 'results/' + 'batch_output.xlsx')
    print('ANALYSIS WAS FINISHED')