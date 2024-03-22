from scipy.io import wavfile
import cv2
import numpy as np

def read_audio(args):
	audio_fp = args.audioFilePath
	_, audio = wavfile.read(audio_fp)
	return audio

def read_video(flist, shot):
	start_shot, end_shot = shot
	start_frame = start_shot.frame_num
	end_frame = end_shot.frame_num
	upper_bound = min(end_frame, len(flist))
	frame = cv2.imread(flist[start_frame])
	all_frames = np.zeros((upper_bound - start_frame, *frame.shape), dtype='uint8')
	all_frames[0] = frame
	for fidx in range(1, upper_bound - start_frame):
		fname = flist[start_frame + fidx]
		frame = cv2.imread(fname)
		all_frames[fidx] = frame
	return all_frames
