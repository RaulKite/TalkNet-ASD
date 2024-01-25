## TalkNet-ASD optimised version

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
