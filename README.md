# A review of image based automatic facial landmark identification techniques
#### Ben Johnston & Philip de Chazal

This github repository contains the source code for the survey article:<br />
<< insert reference here >>

## System Requirements
The source code for this repository was executed and run on a Lenovo W530 laptop with 8GB RAM and a Dell PowerEdge R630 &
C6320 Server nodes with 128GB RAM available.  All of the scripts contained within Sec2_Dataset_selection and
Sec3_FeatureExtraction folders are able to be executed on a desktop computer with a reasonable amount of RAM installed. The
memory requirements of the experiments contained within Sec4_ModelDefinition is significantly higher and required the
use of the Dell Server nodes.  For more information regarding the memory requirements see the journal article. 

## Structure of the repository 
The repository is structured as per the following:

FacialLandmarkingReview<br />
├── experiments<br />
│   ├── Sec2_Dataset_selection<br />
│   │   └── raw_data<br />
│   ├── Sec3_FeatureExtraction<br />
│   │   ├── docker<br />
│   └── Sec4_ModelDefinition<br />
│       └── docker<br />
└── tests<br />

where all of the source code for the survey article is contained within the *experiments* folder divided into the
appropriate sections.  The folders *experiments/Sec3_FeatureExtraction* and *experiments/Sec4_ModelDefinition* also contain
subfolders with the docker build files for the docker images described below.

## Data source location
In order to replicate the reported experimental results the source code requires the [BioID](https://www.bioid.com/About/BioID-Face-Database), [HELEN](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/), [MUCT](https://github.com/StephenMilborrow/muct),
[300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/) and [Menpo](https://ibug.doc.ic.ac.uk/resources/2nd-facial-landmark-tracking-competition-menpo-ben/) datasets.  The location of each of the datasets can be configured using the environment
variables: `BIOID_DATA`, `IBUG_DATA`, `MUCT_DATA`, `MENPO_DATA`.  If environment variables are not provided the
following folder structure is assumed:

~/datasets<br />
├── bioid<br />
│   ├── faces<br />
│   └── points_20<br />
├── BioID<br />
│   ├── faces<br />
│   └── points_20<br />
├── HELEN<br />
│   └── trainset<br />
├── ibug<br />
│   ├── 300W<br />
│   │   ├── 01_Indoor<br />
│   │   ├── 02_Outdoor<br />
│   │   └── combined<br />
│   └── menpo_2017_trainset<br />
└── muct<br />
    └── muct-images <br />

the Python script `./experiments/download_data.py` can be used to both download the BioID and MUCT datasets as well as
create the folder structure as described above once the other datasets have been independently downloaded.

## Executing the code
### via Docker
The easiest way to execute the code is to use the docker images.  If you are new to docker and need some help getting
started we suggest the [getting started](https://docs.docker.com/get-started/) of the documentation.

There are two separate docker images as the environment requirements of Sec2_DatasetSelection and Sec3_FeatureExtraction
differ slightly from Sec4_ModelDefinition.  With docker installed you can download the source code and all environment
requirements simply:

```bash
docker pull debrown/facelmrkreviewsec23 # For Sections 2 and 3
docker pull debrown/facelmrkreviewsec3 # For section 4
```

Once the images have been downloaded they can be run using `docker run`, it is recommended that you share the folder
being used to store the data with the image using the following:

```bash
docker run --rm -v -v /path/to/data/folder:/home/doc-E-brown/datasets -ti debrown/facelmrkreviewsec23 # For Sections 2 and 3
docker run --rm -v -v /path/to/data/folder:/home/doc-E-brown/datasets -ti debrown/facelmrkreviewsec4 # For Section 4
```

The above commands start the docker containers interactively into bash with the corresponding virtual environment
already activated.  If you haven't already downloaded the required datasets you will need to run `~/FacialLandmarkingReview/experiments/download_data.py` you can use the `--help` flag for help.  Once the data is downloaded, simply cd into the appropriate folder `cd ~/FacialLandmarkingReview/experiments/Sec2_DatasetSelection`, `cd ~/FacialLandmarkingReview/experiments/Sec3_FeatureExtraction` or `cd ~/FacialLandmarkingReview/experiments/Sec3_FeatureExtraction` and execute the desired python script. 

### via Source
If you would like to execute the source code directly without the docker images you can download the source code from
this github repo.  In order to use the code you will need to do the following:

**For Sections 1 & 2**

1. Download and extract the source
2. Install Python 3.5 or later
3. Download, build and install opencv (make sure it is compiled for Python 3)
4. Add the opencv libraries to the PYTHON_PATH (it is highly recommended that you use a virtual enviornment) 
5. Execute `cd ./FacialLandmarkingReview/experiments/Sec3_FeatureExtraction && pip install -r requirements.txt`
6. Ensure the data is available as per `./FacialLandmarkingReview/experiments/download_data.py`

**For Section 4**

1. Download and extract the source
2. Install the Python 2 version of conda as per [getting-started](https://conda.io/docs/user-guide/getting-started.html)
3. Create a conda environment `conda create -n FacialLandmarkingReview`
4. If not already activated, activate the environment `source activate FacialLandmarkingReview`
5. Install the dependencies `conda install -c menpo menpofit`
6. Ensure the data is available as per `./FacialLandmarkingReview/experiments/download_data.py`
