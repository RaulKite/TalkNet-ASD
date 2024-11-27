## TalkNet-ASD optimised version

## Standalone run

### Requirements

See the list of required Python packages <details><summary> here </summary>

* facenet-pytorch==2.5.3
* gdown==5.2.0
* numpy==2.0.2
* opencv-python==4.10.0.84
* pandas==2.2.3
* scenedetect==0.6.5
* scikit-learn==1.5.2
* scipy==1.13.1
* torch==2.5.1
* torchvision==0.20.1
* python_speech_features==0.6
* tqdm==4.67.1

</details>

### Scenes

The script requires a list of scenes, which can be produced using ```[pyscenedetect](https://www.scenedetect.com/)```.
Install ```pyscenedetect``` using pip:

> python -m pip install scenedetect

Then run ```pyscenedetect``` with the default parameters for a video to process:

> scenedetect -i $VIDEO_PATH list-scenes -f $PATH_TO_SCENES

where ```$VIDEO_PATH``` is the path to the video and ```$PATH_TO_SCENES``` is the path to the output csv file where the list of scenes will be stored.

### Preprocessing

In the following ```$VIDEO_PATH``` is the path to the video we want to process, ```THREADS=10```, and ```$TMPDIR``` is the path to the output folder.

1. Create the folders required by the original pipeline:

>  mkdir "$TMPDIR"/{pyavi,pyframes,pywork}

2. Convert the video to 25 fps ratio:

> ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -async 1 -r 25 $TMPDIR/temp_video_25fps.avi -loglevel panic

3. Extract the audio and convert it to a 16kHz wav file::

> ffmpeg -y -i $VIDEO_PATH -qscale:a 0 -ac 1 -vn -threads $THREADS -ar 16000 $TMPDIR/temp_audio_16khz.wav -loglevel panic

4. Extract frames from the converted video:

> ffmpeg -y -i $TMPDIR/temp_video_25fps.avi -qscale:v 2 -threads $THREADS -f image2 $TMPDIR/pyframes/%07d.jpg -loglevel panic

### Running the script

1. Clone this repository:

> git clone git@github.com:iburenko/TalkNet-ASD.git

Suppose you cloned it to ```$TALKNET_REPO_PATH```.

2. Change your working directory to ```$TALKNET_REPO_PATH```:

> cd "$TALKNET_REPO_PATH"

3. Finally, run the script:

Set ```$DEVICE``` either to ```cuda``` or ```cpu``` accordingly to the availablity of GPUs.

> python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TMPDIR \
    --audioFilePath $TMPDIR/temp_audio_16khz.wav \
    --device $DEVICE

Add ```--visualisation``` if you want to create a video file with bounding boxes for all the detected tracks:

> python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TMPDIR \
    --audioFilePath $TMPDIR/temp_audio_16khz.wav \
    --device $DEVICE \
    --visualisation

### Output

The main output of the pipeline are two binary files stored in ```$TMPDIR/pywork```:

1. ```scores.pckl``` -- stores scores for each track. The score reflects how likely it is that the person from the track speaks. By default, positive scores are assigned to active speakers. The content of the file is the list of numpy arrays.
2. ```trakcs.pckl`` -- stores all the detected tracks. Roughly, one track corresponds to one person detected in a scene. The file stores a list of Python dictionaries. Each dictionary has the following keys:
    * ```track```. Keeps information about the current track in the form of Python dictionary with two keys:
        * ```frame``` -- numpy array of frames (in 25 fps, sic!).
        * ```bbox``` -- numpy array of bounding boxes for a face.
    * ```proc_track```
    * ```embeddings``` -- numpy array of face embeddings for each face from the track.

## Run on HPC

Use one of bash scripts to run the pipeline:
```
sbatch run_asd_for_video_{alex,fritz}.sh $CASE_ID [--visualisation]
```
```--visualisation``` is an optional parameter. If given, the script will 
create a visualisation.

By default the results (scores, bounding boxes' coordinates, etc.) will be
copied to
```
/home/atuin/b105dc/data/datasets/russian_propaganda_dataset_openpose/test_asd/$CASE_ID
```

### Bash scripts
- ```run_asd_for_video_alex.sh``` will submit a task on Alex using 1 gpu;
- ```run_asd_for_video_fritz.sh``` will submit a task on Fritz using cpus;

* A python environment will be created adn activated from a script.
All the required packages will be installed.
