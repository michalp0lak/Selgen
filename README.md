# Selgen software for

## Authors
Michal Polak [1], Jan Humplik [1]

[1] Palacky Univ, Fac Sci, Dept Chemical Biology & Genetics, Olomouc 77900, Czech Republic

## Description
The software is designed to evaluate batch of images generated with Selgen experimental design. Core of the software is mainly using methods of OpenCV library.

Single image contains two trays of 10x8 holes. This is region of interest (ROI) for further analysis.

The goal is to evaluate spatial and color pattern of plant in each cell. For experiment "green" pixels are valid. What is "green" can be defined by user in **global_variables.py** as a thresholds for segmentation.

Software suppose various image formats jpg, png, bmp, tiff, tif as an input data.

The output of analysis is **xlsx** file described below and folder of raw images with painted contours around "green" pixels.

Process of single image analysis follows these steps:
1. ROI is cropped from raw image
2. Trays are separated into two ROIs (left and right)
3. Mask of tray grid is segmented in each ROI
4. Computation of grid for both trays
5. Separation of trays into individual holes
6. Computation of spatial and color pattern in each hole


### Getting started
These instructions will lead you to copy project and deeploy on your local machine for your use. 

### Platform
Project is developed in **Python 3** programming language. At this moment I expect that analysis will be executed on server with *Windows10* OS. So at this moments all instructions stand for *Windows10* OS.

### Prerequisities
1. As a first install Python up to date version (not older than **3.8.X**) from the Microsoft Store. Or download python [on python official website](https://www.python.org/downloads/) and install python by following instructions for example [on this website](https://realpython.com/installing-python/#windows).

### Deployment
1. As a first python **virtualenv** package is required. Open *command-line interpreter (cmd.exe)* terminal and execute `py -m pip install --user virtualenv`
2. Download zipped selgen project from github repository [https://github.com/PolakMichalMLT/Selgen](https://github.com/PolakMichalMLT/Selgen) with green button *Clone or download* and *Download ZIP* option.
3. Unzip downloaded file **Selgen-master.zip** and put **Selgen-master** folder on your desktop
4. In *cmd* terminal navigate terminal into this folder `cd .\Desktop\Selgen-master`
5. Create python virtual enviroment with command `python -m venv selgen`
6. Activate virtual enviroment with `.\selgen\Scripts\activate`
7. Install project requirements `pip install -r requirements.txt`

### Execution of analysis
1. Create folder with images of experiment for processing
2. Setup project **global variables**. In folder *your/project/path/selgen/Lib/site-packages/Selgen-master* open file **selgen_global.py** in some text editor and define:
   - *path* as directory of folder from *step 1*
3. Navigate *cmd* terminal to *your\project\path\selgenvir\Lib\site-packages\Selgen-master*
4. In *cmd* terminal execute analysis with `python selgen_analysis.py`

#### Output of analysis
- all results are located in *path* from **selgen_global.py** file
- in *contoured_images* folder are original images with drawed contours of active biomass
- in *processed* folder are images, which were sucessfully processed
- in *unprocessed* folder are images, which were not processed because of som e error
- in *batch_output.xlsx* are structured results of evaluated batch
  - **biomass** column is value of statistic = segmented pixels of plant in given area
  - **day** column is number of experiment day
  - **side** column specify tray side
  - **variant** column specify treatment/variant
  - **row** column indicates row in tray grid of evaluated area
  - **column** column indicates column in tray grid of evaluated area
  - **size** is size of evaluated area

  
## Obtaining the software, technical and legal terms:
The software is written in Python.
 
The software can be used without any charge upon obtaining license from the author.
 
The licence can be obtained by e-mail upon agreeing not to use the software for commercial purpose.

After obtaining the license, the end-user will be provided (free of charge) with the link of Github repository of the project.

To obtain the files, please contact Michal Polak by email michal.polak@upol.cz.

Your email must contain the following statement:

"Software will not be used for any commercial purpose."
