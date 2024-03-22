from scenedetect.frame_timecode import  FrameTimecode

def correct_timecode(timecode):
    H, M, S_MS = timecode.split(":")
    S, MS = S_MS.split(".")
    if S == "60":
        S = "00"
        M = str(int(M) + 1).zfill(2)
        timecode = H + ":" + M + ":" + S + "." + MS
    return timecode

def split_long_scene(scene, scene_len, fps=25):
	augmented_scenes = 0
	augmented_scene_list = list()
	one_minute = FrameTimecode("00:01:00.000", fps)
	while (augmented_scenes + 1) * 60 < scene_len:
		if augmented_scenes == 0:
			prev_scene_boundary = scene[0]
		else:
			prev_scene_boundary = augmented_scene_list[-1][1]
		new_scene_boundary = prev_scene_boundary + one_minute
		augmented_scene_list.append((prev_scene_boundary, new_scene_boundary))
		augmented_scenes += 1
	augmented_scene_list.append((augmented_scene_list[-1][1], scene[1]))
	return augmented_scene_list

def split_all_all_scenes(all_scenes):
	all_adjusted_scenes = list()
	for scene in all_scenes:
		scene_len = scene[1].get_seconds() - scene[0].get_seconds()
		if scene_len > 60:
			scene_list = split_long_scene(scene, scene_len)
		else:
			scene_list = [scene]
		all_adjusted_scenes.extend(scene_list)
	return all_adjusted_scenes

def read_scenes(args, fps=25):
	with open(args.pathToScenes, "r") as file:
		all_lines = file.readlines()
	all_scenes = list()
	for line in all_lines[2:]:
		splitted_line = line.split(',')
		_, _, start_timecode, _, _, end_timecode, *_ = splitted_line
		start_timecode = correct_timecode(start_timecode)
		end_timecode = correct_timecode(end_timecode)
		start_frame = FrameTimecode(start_timecode, fps)
		end_frame = FrameTimecode(end_timecode, fps)
		all_scenes.append((start_frame, end_frame))
	all_adjusted_scenes = split_all_all_scenes(all_scenes)
	return all_adjusted_scenes