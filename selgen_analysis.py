import pandas as pd
import cv2
import numpy as np
import scipy.ndimage
import scipy.io
from scipy.signal import argrelmax
from scipy.optimize import curve_fit
import os, sys
import selgen_global
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

def crop_ROI(image):

    #crop_ROI function localize region of interest = black tray with crop .
    #Localization is based on gradient between white background and black borders of tray 
    
    assert (type(image) == np.ndarray) & (len(image.shape) == 3) & np.amin(image) >= 0 & np.amax(image) <= 255, 'Input data has to be RGB image'
    
    try:

        b,g,r = cv2.split(image)

        #columns
        col_mask = b < 100
        col_mask[:,col_mask.shape[1]//2 - 500:col_mask.shape[1]//2 + 500] = 1

        proj = np.sum(col_mask, axis = 0)
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

        row_mask = b < 100

        proj = np.sum(row_mask,axis = 1)
        mez = max(proj)/3
        pos = round(len(proj)/2)

        while proj[pos] > mez:
            pos = pos+1

        dor = pos

        pos = round(len(proj)/2)

        while proj[pos]>mez:
            pos = pos-1

        odr = pos

        ROI = image[odr:dor,ods:dos,:]
        
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
        
        return grid_mask * background_mask

    except Exception as e:

        raise e


def find_opt_rotation(mask: np.ndarray):

    (h, w) = mask.shape
    (cX, cY) = (w // 2, h // 2)

    Crit = []
    Angle = []

    for angle in np.arange(-20,21):

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(mask, M, (w, h))

        Crit.append(np.var(np.sum(rotated, axis = 1)) + np.var(np.sum(rotated, axis = 1)))
        Angle.append(angle)

    return Angle[np.argmax(Crit)]


def nufit_fourier(x, y):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''

    x = np.array(x)
    y = np.array(y)

    freq = np.fft.fftfreq(len(x), (x[1]-x[0]))   # assume uniform spacing
    Fy = abs(np.fft.fft(y))

    guess_freq = abs(freq[np.argmax(Fy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c

    popt, pcov = curve_fit(sinfunc, x, y, p0=guess, maxfev = 10000)
    A, w, p, c = popt
    fitfunc = lambda t: A * np.sin(w*t + p) + c

    return fitfunc(x)

def get_grid_coords(grid_mask, side):

    #get_grid_coords identify coordinates of grid on given part of tray with grid mask
        
    assert (type(grid_mask) == np.ndarray) & (len(grid_mask.shape) == 2) & np.amin(grid_mask) >= 0 & np.amax(grid_mask) <= 255, 'Input data has to be RGB image'
    assert side in ('left','right'), 'Side argument is string left or right'

    try:

        #Find optimal grid rotation and rotate grid
        h,w = grid_mask.shape

        grid_mask[0:150,:] = 0
        grid_mask[h-150:h,:] = 0

        if(side == 'left'):

            grid_mask[:,0:200] = 0
            grid_mask[:,w-100:w] = 0

        if(side == 'right'):

            grid_mask[:,0:100] = 0
            grid_mask[:,w-200:w] = 0

        (cX, cY) = (w // 2, h // 2)

        opt_angle = find_opt_rotation(grid_mask)

        M = cv2.getRotationMatrix2D((cX, cY), opt_angle, 1.0)
        grid_mask = cv2.warpAffine(grid_mask, M, (w, h))

        #Compute signal for rows and columns
        row_idx = [] 
        row_signal = []

        for i in range(0,grid_mask.shape[0]):

            row_signal.append(np.sum(grid_mask[i,:]))
            row_idx.append(i)

        row_idx = np.array(row_idx)
        row_signal = np.array(row_signal)

        col_idx = [] 
        col_signal = []

        for j in range(0,grid_mask.shape[1]):

            col_signal.append(np.sum(grid_mask[:,j]))
            col_idx.append(j)

        col_idx = np.array(col_idx)
        col_signal = np.array(col_signal)

        #Using fourier tranform find grid indexes
        row_fit = nufit_fourier(row_idx, row_signal)
        row_indexes = list(argrelmax(row_fit)[0])

        col_fit = nufit_fourier(col_idx, col_signal)
        col_indexes = list(argrelmax(col_fit)[0])

        if len(col_indexes) != 9:

            index_energy = []

            for index in col_indexes: 
                
                indexes = np.arange(index-30,index+30)
                index_energy.append(col_signal[list(indexes[(indexes > 0) & (indexes < w)])].sum())

            if len(col_indexes) < 9:

                if index_energy[0] != 0: col_indexes = [int(np.max([0,col_indexes[0] - np.median(np.diff(col_indexes))]))] + col_indexes
                elif index_energy[-1] != 0: col_indexes = col_indexes + [int(np.min([w,col_indexes[-1] + np.median(np.diff(col_indexes))]))]

            elif len(col_indexes) > 9:

                if side == 'left': col_indexes = col_indexes[-9:]
                elif side == 'right': col_indexes = col_indexes[:9]

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

        if (len(left_part_row) != 6) | (len(right_part_row) != 6) | (len(left_part_col) != 9) | (len(right_part_col) != 9): 
            raise Exception('Grid structure of tray wasn\'t found')

        #print(len(left_part_row), len(right_part_row),len(left_part_col),len(right_part_col))

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

def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]



def process_images(imageLoad):
    
    print("[INFO] Starting process {}".format(imageLoad["id"]))
    print("[INFO] For process {}, there is {} images in processing queue".format(imageLoad["id"], len(imageLoad["files_names"])))
    
    f = open(imageLoad["temp_path"] + "failures_{}.txt".format(imageLoad["id"]),"w+")
    final_data = []

    for imageName in imageLoad["files_names"]:
        
        try:
            
            metadata = filename_parser(imageName)

            image = cv2.imread(imageLoad["input_path"] + imageName)
            
            areas, roi = process_image(image)

            contoured_roi = paint_active_biomass(roi, selgen_global.lower_thresh, selgen_global.upper_thresh)

            cv2.imwrite(imageLoad["output_path"] + imageName, contoured_roi)

            image_data = []

            for area in areas:
                
                biomass = segmentation_biomass(area.cropped_area, selgen_global.lower_thresh, selgen_global.upper_thresh)

                image_data.append(dict(zip(('filename','date','time','variant','side','row', 'column','biomass', 'size'),
                            (imageName, metadata['date'], metadata['time'], metadata['variant'], area.side, area.row, area.column, biomass, area.size))))    

            final_data = final_data + image_data
           
        except Exception as e:

            exception_type, exception_object, exception_traceback = sys.exc_info()

            filename = exception_traceback.tb_frame.f_code.co_filename

            line_number = exception_traceback.tb_lineno
        
            print('{} - {}: {}'.format(filename, line_number, exception_object))
            
            f.write('{},{filename},{line_numbe},{exception_tracebac}'.format(imageName) + '\n')
    
        df = pd.DataFrame(final_data)
        df = df[['filename','date','time','variant','side','row', 'column','biomass', 'size']]
        df.sort_values(by=['side', 'row', 'column', 'date', 'time'])
        df.to_excel(imageLoad["temp_path"] + '/batch_result_{}.xlsx'.format(imageLoad["id"]))

if __name__ == '__main__':

    assert os.path.exists(selgen_global.path) , 'path should navigate into the folder where batch of images are stored'
    
    temp_path = selgen_global.path + 'temp/'
    output_path = selgen_global.path + 'results/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    formats = ('.JPG','.jpg','.PNG','.png','.bmp','.BMP','.TIFF','.tiff','.TIF','.tif')
    files = [file for file in os.listdir(selgen_global.path) if file.endswith(formats)]
    data = [] 

    procs = cpu_count()
    procIDs = list(range(0, procs))

    numImagesPerProc = len(files) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))
        
    chunkedPaths = list(chunk(files, numImagesPerProc))
    
    # initialize the list of payloads
    imageLoads = []

    # loop over the set chunked image paths
    for (i, fileNames) in enumerate(chunkedPaths):

        # construct a dictionary of data for the payload, then add it
        # to the payloads list
        data = {
            "id": i,
            "files_names": fileNames,
            "input_path": selgen_global.path,
            "output_path": output_path,
            "temp_path": temp_path
        }
        imageLoads.append(data)
        
    structured_data = []
    
    # construct and launch the processing pool
    print("[INFO] Launching pool using {} processes.".format(procs))
    print("[INFO] All CPU capacity is used for data analysis. You won't be able to use your computer for any other actions.")

    pool = Pool(processes=procs)
    pool.map(process_images, imageLoads)
    pool.close()
    pool.join()

    print("[INFO] Pool of processes was closed")
    print("[INFO] Aggregating partial results into structured data set.")

    xlsx_files = [file for file in os.listdir(temp_path) if file.endswith('xlsx')]
    txt_files = [file for file in os.listdir(temp_path) if file.endswith('txt')]

    frames = []

    for xlsx in xlsx_files:
        
        frames.append(pd.read_excel(temp_path + xlsx, engine='openpyxl'))
        
    structured_result = pd.concat(frames, ignore_index=True)
    structured_result = structured_result[['filename','date','time','variant','side','row', 'column','biomass', 'size']]
    structured_result.sort_values(by=['side', 'row', 'column', 'date', 'time'])
    structured_result.to_excel(output_path + 'exp_result.xlsx', index=False)


    with open(output_path + 'failures.txt', 'w') as outfile:
        for fname in txt_files:
            with open(temp_path+fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
    files = [file for file in os.listdir(temp_path)]
                                            
    for f in files:
        os.remove(temp_path+f)
        
    os.rmdir(temp_path)

    print("[INFO] ANALYSIS WAS FINISHED")