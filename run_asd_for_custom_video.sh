#!/bin/bash -l

#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBACTH --export=NONE
#SBATCH --job-name=for_raul
#SBATCH --output /home/atuin/b105dc/data/datasets/talknet_examples_for_raul/logs/asd_on_%x_job_id_%j.out

unset SLURM_EXPORT_ENV

export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'

# Get CL parameter
VIDEO_PATH=$1
CASE_ID=$(basename $VIDEO_PATH)
CASE_ID="${CASE_ID%.*}"
SAVE_TO="/home/atuin/b105dc/data/datasets/talknet_examples_for_raul/output/$CASE_ID"
DEVICE=cuda

# Create output folder
if [ ! -d "$SAVE_TO" ]; then
    mkdir -p $SAVE_TO
fi

# Define paths
module load ffmpeg

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


# ffmpeg=/home/atuin/b105dc/data/software/ffmpeg/ffmpeg
PATH_TO_SCENES="$SAVE_TO/$CASE_ID.csv"
FPS=25
scenedetect -i "$VIDEO_PATH" --framerate "$FPS" list-scenes --filename "$PATH_TO_SCENES" --quiet
THREADS=10

echo $CASE_ID
echo "VIDEO PATH = "$VIDEO_PATH
echo "SAVE TO "$SAVE_TO

# If file exsists, exit
if [ -f $SAVE_TO/"tracks.pckl" ]; then
    echo "File exists! Exiting..."
    exit 1
fi


mkdir $TMPDIR/{pyavi,pyframes,pywork}

ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -async 1 -r 25 $TMPDIR/temp_video_25fps.avi -loglevel panic
ffmpeg -y -i $VIDEO_PATH -qscale:a 0 -ac 1 -vn -threads $THREADS -ar 16000 $TMPDIR/temp_audio_16khz.wav -loglevel panic
ffmpeg -y -i $TMPDIR/temp_video_25fps.avi -qscale:v 2 -threads $THREADS -f image2 $TMPDIR/pyframes/%07d.jpg -loglevel panic

cd /home/hpc/b105dc/b105dc10/TalkNet-ASD
python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TMPDIR \
    --audioFilePath $TMPDIR/temp_audio_16khz.wav \
    --device $DEVICE \
    --visualisation

cp $TMPDIR/pywork/* $SAVE_TO/
if [ -f $TMPDIR/pyavi/video_out.avi ]; then 
    cp $TMPDIR/pyavi/video_out.avi $SAVE_TO/$CASE_ID"_"$DEVICE".avi"
fi
