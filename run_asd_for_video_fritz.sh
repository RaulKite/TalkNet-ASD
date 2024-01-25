#!/bin/bash -l

#SBATCH --partition=singlenode
#SBATCH --time=08:00:00
#SBATCH --job-name=talknet_fritz
#SBACTH --export=NONE

unset SLURM_EXPORT_ENV

RUN_FOLDER=$(pwd)

module load python
cd /tmp/$SLURM_JOB_ID.fritz
python -m venv venvs/foo
source venvs/foo/bin/activate
python -m pip install gdown scikit-learn
python -m pip install torch numpy scipy python_speech_features opencv-python facenet-pytorch pandas tqdm --quiet
python -m pip install --upgrade scenedetect[opencv] --quiet
cd $RUN_FOLDER

ffmpeg=/home/atuin/b105dc/data/software/ffmpeg/ffmpeg
ALL_VIDEOS_LIST=/home/atuin/b105dc/data/datasets/russian_propaganda_dataset_openpose/all_russian_propaganda_dataset.txt
ALL_SCENES_FOLDER=/home/atuin/b105dc/data/datasets/russian_propaganda_dataset_openpose/scenes
THREADS=10

CASE_ID=$1
DEVICE=cpu

VIDEO_PATH=$(cat $ALL_VIDEOS_LIST | grep $CASE_ID)
SAVE_TO=$DATASET/russian_propaganda_dataset_openpose/test_asd/$CASE_ID
PATH_TO_SCENES=$ALL_SCENES_FOLDER/$CASE_ID-Scenes.csv

mkdir $TMPDIR/{pyavi,pyframes,pywork}
mkdir -p $SAVE_TO

$ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -async 1 -r 25 $TMPDIR/temp_video_25fps.avi -loglevel panic
$ffmpeg -y -i $VIDEO_PATH -qscale:a 0 -ac 1 -vn -threads $THREADS -ar 16000 $TMPDIR/temp_audio_16khz.wav -loglevel panic
$ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -f image2 $TMPDIR/pyframes/%07d.jpg -loglevel panic

cd /home/atuin/b105dc/data/work/iburenko/talknet_optimised/TalkNet-ASD
python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TMPDIR \
    --audioFilePath $TMPDIR/temp_audio_16khz.wav \
    --device $DEVICE 

cp $TMPDIR/pywork/* $SAVE_TO/
if [ -f $TMPDIR/pyavi/video_out.avi ]; then 
    cp $TMPDIR/pyavi/video_out.avi $SAVE_TO/$CASE_ID"_"$DEVICE".avi"
fi