# Selgen software for crop cold resistance analysis

## Description
The software is designed to evaluate batch of images generated with experimental design of Selgen company. The software is coded in Python3 and mainly uses OpenCV library.

Single image contains two trays (left and right) of 5x8 growing areas. This is region of interest (ROI) for further analysis.

The goal is to evaluate spatial and color pattern of plant in each cell. For experiment "green" pixels are valid. What is "green" can be defined by user in **global_variables.py** as a thresholds for segmentation.

Software suppose various image formats jpg, png, bmp, tiff, tif as an input data.

The output of analysis is **xlsx** file described below and folder of raw images with painted trays grid and contours around crop.

## Processing pipeline

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/raw.png?raw=true)

Process of single image analysis follows these steps:

1. ROI is cropped from raw image

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/roi.png?raw=true)

2. Tray is splitted into 2 ROI areas (left and right)

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/split.png?raw=true)

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/left_part.png?raw=true)
![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/right_part.png?raw=true)

3. Mask of tray grid is segmented in each ROI

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/left_part_mask.png?raw=true)
![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/right_part_mask.png?raw=true)

4. Computation of ROI mask optimal rotation

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/left_part_mask_rot.png?raw=true)

5. Localization of ROI grid with fourier transform

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/fourier.png?raw=true)

6. ROI separation into growing areas

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/left_part_mask_grid.png?raw=true)

7. Computation of spatial and color pattern in each growing area

![alt text](https://github.com/PolakMichalMLT/Selgen//blob/master/readme_images/final.png?raw=true)

### Deployment

#### Unix
1. Python virtualenv package is required. Open terminal and execute `python3 -m pip install virtualenv` command to install this package.
2. Download zipped selgen project from github repository [https://github.com/UPOL-Plant-phenotyping-research-group/Selgen](https://github.com/UPOL-Plant-phenotyping-research-group/Selgen) with green button *Clone or download* and *Download ZIP* option.
3. Open downloaded zip file and place *Selgen-master* folder on Desktop of your server.
4. Open terminal and navigate into project folder with command `cd 'your_path_to_project' (e.g. ...../Desktop/Selgen-master/)`.
5. In terminal execute command `bash create_venv`, which will create virtual enviroment. Now in project folder *...../Desktop/Selgen-master/* you should find *selgen* folder.

#### Windows
1. As a first python **virtualenv** package is required. Open *command-line interpreter (cmd.exe)* terminal and execute `py -m pip install --user virtualenv`
2. Download zipped selgen project from github repository [https://github.com/PolakMichalMLT/Selgen](https://github.com/PolakMichalMLT/Selgen) with green button *Clone or download* and *Download ZIP* option.
3. Open downloaded zip file and place *Selgen-master* folder on Desktop of your server.
4. In *cmd* terminal navigate terminal into this folder `cd .\Desktop\Selgen-master`
5. Create python virtual enviroment with command `python -m venv selgen`
6. Activate virtual enviroment with `.\selgen\Scripts\activate`
7. Install project requirements `pip install -r requirements.txt`
8. Deactivate virtual enviroment with `.\selgen\Scripts\deactivate`

### Execution of analysis

#### Unix
1. Create folder with images of experiment for processing.
2. In project folder *....../Desktop/Selgen-master* open file **selgen_global.py** in some text editor (Notepad++) and define:
   - *path* as directory of folder from *step 1*
3. Navigate terminal to project folder *...../Desktop/Selgen-master*.
4. In terminal execute command `bash exe` which will execute data analysis.

#### Windows
1. Create folder with images of experiment for processing
2. In folder *....../Desktop/Selgen-master* open file **selgen_global.py** in some text editor (Notepad++ and define:
   - *path* as directory of folder from *step 1*
3. Navigate *cmd* terminal to *....../Desktop/Selgen-master* 
4. Activate virtual enviroment with `.\selgen\Scripts\activate`
5. In *cmd* terminal execute analysis with `python selgen_analysis.py`
6. Deactivate virtual enviroment with `.\selgen\Scripts\deactivate`

### Output of analysis
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
