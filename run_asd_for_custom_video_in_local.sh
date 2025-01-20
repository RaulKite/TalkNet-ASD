export LANG='en_US.UTF-8'
export LC_ALL='en_US.UTF-8'

# Get CL parameter
VIDEO_PATH=$1
CASE_ID=$(basename $VIDEO_PATH)
CASE_ID="${CASE_ID%.mp4}"
echo "**************************************************$CASE_ID**************************************************"

SAVE_TO="./output/$CASE_ID"
DEVICE=cuda

# Create output folder
if [ ! -d "$SAVE_TO" ]; then
    mkdir -p $SAVE_TO
fi

# Create temporary directories
TEMP_DIR="/tmp/talknet_processing"
mkdir -p $TEMP_DIR/{pyavi,pyframes,pywork}

# Define paths and parameters
PATH_TO_SCENES="$SAVE_TO/$CASE_ID.csv"
FPS=25
THREADS=10

# Print main variables
echo "Processing video with following parameters:"
echo "CASE_ID = $CASE_ID"
echo "VIDEO_PATH = $VIDEO_PATH" 
echo "SAVE_TO = $SAVE_TO"
echo "TEMP_DIR = $TEMP_DIR"
echo "PATH_TO_SCENES = $PATH_TO_SCENES"
echo "FPS = $FPS"
echo "THREADS = $THREADS"
echo "DEVICE = $DEVICE"



# Run scene detection
scenedetect -i "$VIDEO_PATH" --framerate "$FPS" list-scenes --filename "$PATH_TO_SCENES" --quiet

echo $CASE_ID
echo "VIDEO PATH = "$VIDEO_PATH
echo "SAVE TO "$SAVE_TO

# If file exists, exit
if [ -f $SAVE_TO/"tracks.pckl" ]; then
    echo "File exists! Exiting..."
    exit 1
fi

# Process video and audio
ffmpeg -y -i $VIDEO_PATH -qscale:v 2 -threads $THREADS -async 1 -r 25 $TEMP_DIR/temp_video_25fps.avi -loglevel panic
ffmpeg -y -i $VIDEO_PATH -qscale:a 0 -ac 1 -vn -threads $THREADS -ar 16000 $TEMP_DIR/temp_audio_16khz.wav -loglevel panic
ffmpeg -y -i $TEMP_DIR/temp_video_25fps.avi -qscale:v 2 -threads $THREADS -f image2 $TEMP_DIR/pyframes/%07d.jpg -loglevel panic

# Run TalkNet
cd /data/home/raul/miscosas/repos/TalkNet-ASD  # Make sure to update this path
python run_talknet.py \
    --pathToScenes $PATH_TO_SCENES \
    --videoName temp_video_25fps \
    --videoFolder $TEMP_DIR \
    --audioFilePath $TEMP_DIR/temp_audio_16khz.wav \
    --device $DEVICE \
    --visualisation

# Copy results
cp $TEMP_DIR/pywork/* $SAVE_TO/
if [ -f $TEMP_DIR/pyavi/video_out.avi ]; then 
    cp $TEMP_DIR/pyavi/video_out.avi $SAVE_TO/$CASE_ID"_"$DEVICE".avi"
fi

# Cleanup
rm -rf $TEMP_DIR
