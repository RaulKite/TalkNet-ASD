#!/bin/bash -l

#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBACTH --export=NONE
#SBATCH --output /home/atuin/b105dc/data/work/iburenko/gesture-tokenizer-work/logs/asd_on_rpd_video_id_%x_job_id_%j.out

unset SLURM_EXPORT_ENV

export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'

function get_full_saveto_path () {
    VIDEO_PATH=$(grep -- "$CASE_ID" "$ALL_VIDEOS_LIST" | grep -v f244)
    PATH_SUFFIX=$(echo $VIDEO_PATH | cut -d '/' -f 7-9)
    SAVE_TO=$DATASET/russian_propaganda_dataset_openpose/asd_output/$PATH_SUFFIX/$CASE_ID
}

# Get CL parameter
CASE_ID=$1
DEVICE=cuda

# Define paths
ffmpeg=/home/atuin/b105dc/data/software/ffmpeg/ffmpeg
ALL_VIDEOS_LIST=/home/hpc/b105dc/b105dc10/gesture-tokenizer/all_video_files_russian_propaganda_dataset.txt
ALL_SCENES_FOLDER=/home/atuin/b105dc/data/datasets/russian_propaganda_dataset_openpose/scenes
PATH_TO_SCENES=$ALL_SCENES_FOLDER/$CASE_ID-Scenes.csv
THREADS=10


echo $CASE_ID
get_full_saveto_path
echo "VIDEO PATH = "$VIDEO_PATH
echo "SAVE TO "$SAVE_TO

# If file exsists, exit
if [ -f $SAVE_TO/"tracks.pckl" ]; then
    echo "File exists! Exiting..."
    exit 1
fi

# Create python environment
RUN_FOLDER=$(pwd)
module load python
cd /tmp/$SLURM_JOB_ID.alex # $TMPDIR
python -m venv venvs/foo
source venvs/foo/bin/activate
python -m pip install gdown scikit-learn --quiet
python -m pip install torch numpy scipy python_speech_features opencv-python facenet-pytorch pandas tqdm --quiet
python -m pip install --upgrade scenedetect[opencv] --quiet
cd $RUN_FOLDER


mkdir $TMPDIR/{pyavi,pyframes,pywork}

$ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -async 1 -r 25 $TMPDIR/temp_video_25fps.avi -loglevel panic
$ffmpeg -y -i $VIDEO_PATH -qscale:a 0 -ac 1 -vn -threads $THREADS -ar 16000 $TMPDIR/temp_audio_16khz.wav -loglevel panic
$ffmpeg -y -i $TMPDIR/temp_video_25fps.avi -qscale:v 2 -threads $THREADS -f image2 $TMPDIR/pyframes/%07d.jpg -loglevel panic

cd /home/hpc/b105dc/b105dc10/TalkNet-ASD
python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TMPDIR \
    --audioFilePath $TMPDIR/temp_audio_16khz.wav \
    --device $DEVICE

# Create output folder
if [ ! -d "$SAVE_TO" ]; then
    mkdir -p $SAVE_TO
fi

cp $TMPDIR/pywork/* $SAVE_TO/
if [ -f $TMPDIR/pyavi/video_out.avi ]; then 
    cp $TMPDIR/pyavi/video_out.avi $SAVE_TO/$CASE_ID"_"$DEVICE".avi"
fi
