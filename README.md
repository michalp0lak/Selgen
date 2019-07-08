# Selgen
Project was created by Michal Polak and Jan Humplik, as employees of **Center of the Region Hana of Palacky University in Olomouc** for **Selgen** company. Purpose of project code is evaluation of **Selgen** experimental data. Data have form of RGB images. Camera monitors trays with plant, these trays are analyzed objects (ROI). In experiments plant recovery after stress exposion is evaluated.
## Platform
Project is developed only in **Python 3** programming language. At this moment I expect that analysis will be executed on server with *Windows10* OS. So at this moments all instructions stand for *Windows10* OS.
## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. Project is not prepared for deployment in live system.
### Prerequisities
1. As a first download python version **3.6.X** [on python official website](https://www.python.org/downloads/) and install python by following instructions for example [on this website](https://realpython.com/installing-python/#windows)
2. Download *Anaconda Data Science platform* from [official website](https://www.anaconda.com/distribution/#windows), Python **3.X** version and follow installation instructions [https://docs.anaconda.com/anaconda/install/windows/](https://docs.anaconda.com/anaconda/install/windows/)
### Installing
1. As a first python **virtualenv** package is needed. Open *Anaconda prompt* terminal and execute `pip install virtualenv`
2. Continue with *Anaconda prompt* terminal and create folder were your project will be located, execute command `mkdir your/project/path/....` and navigate terminal into this folder `cd your/project/path/....`
3. Create python virtual enviroment with command `virtualenv selgenvir`
4. Download zipped selgen project from github repository [https://github.com/PolakMichalMLT/Selgen](https://github.com/PolakMichalMLT/Selgen) with green button *Clone or download* and *Download ZIP* option.
5. Unzip downloaded zip file to *your/project/path/selgenvir/Lib/site-packages/*
6. Navigate *Anaconda prompt* terminal to project folder with `cd your/project/path/....`. Activate virtual enviroment with `selgenvir\Scripts\activate`
7. Navigate *Anaconda prompt* terminal to *your/project/path/selgenvir/Lib/site-packages/Selgen-master* folder. Install project requirements `pip install selgen_requirements.txt`
## Execution of analysis
1. Create folder with images from experiment for processing
2. Setup project **global variables**. In folder *your/project/path/selgenvir/Lib/site-packages/Selgen-master* open file **selgen_global.py** in some text editor and define:
   - *path* as directory of folder from *step 1*
   - *etalon_path* as *your/project/path/selgenvir/Lib/site-packages/Selgen-master/etalon.mat*
3. Navigate *Anaconda prompt* terminal to *your/project/path/selgenvir/Lib/site-packages/Selgen-master*
4. In *Anaconda prompt* terminal execute analysis with `python selgen_execution.py`
### Output of analysis
- all results are located in *path* from **selgen_global.py** file
- in *contoured_images* folder are original images with drawed contours of active biomass
- in *processed* folder are images, which were sucessfully processed
- in *unprocessed* folder are images, which were not processed because of som e error
- in *batch_output.xlsx* are structured results of evaluated batch
  - **biomass** column is value of statistic = segmented pixels of plant in given area
  - **day** column is number of experiment day
  - **side** column specify tray side
  - **location** column indicates location in given tray side
  - **variant** column specify treatment/variant
  ## Analysis description
