import os, argparse, glob, subprocess, warnings, pickle, math
import copy
from time import time

import python_speech_features
import cv2
import torch
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
import torchvision
from facenet_pytorch import InceptionResnetV1

from model.faceDetector.s3fd import S3FD
from talkNet import talkNet
from io_utils import read_audio, read_video
from scene_utils import read_scenes

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description = "TalkNet Demo or Columnbia ASD Evaluation")

parser.add_argument('--pathToScenes', type=str)
parser.add_argument('--audioFilePath', type=str)
parser.add_argument('--videoName',             type=str, default="3ol_A4gmF3g",   help='Demo video name')
parser.add_argument('--videoFolder',           type=str, default="demo",  help='Path for inputs, tmps and outputs')
parser.add_argument('--pretrainModel',         type=str, default="pretrain_TalkSet.model",   help='Path for the pretrained TalkNet model')

parser.add_argument('--nDataLoaderThread',     type=int,   default=10,   help='Number of workers')
parser.add_argument('--facedetScale',          type=float, default=1, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
parser.add_argument('--minTrack',              type=int,   default=10,   help='Number of min frames for each shot')
parser.add_argument('--numFailedDet',          type=int,   default=10,   help='Number of missed detections allowed before tracking is stopped')
parser.add_argument('--minFaceSize',           type=int,   default=1,    help='Minimum face size in pixels')
parser.add_argument('--cropScale',             type=float, default=0.40, help='Scale bounding box')

parser.add_argument('--start',                 type=int, default=0,   help='The start time of the video')
parser.add_argument('--duration',              type=int, default=0,  help='The duration of the video, when set as 0, will extract the whole video')

parser.add_argument('--device', type=str, choices=["cuda", "cpu"])
parser.add_argument('--visualisation', action='store_true')

args = parser.parse_args()

device = torch.device("cuda" if args.device == "cuda" else "cpu") 

if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
    subprocess.call(cmd, shell=True, stdout=None)

args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
args.savePath = args.videoFolder

FPS = 25

def define_scale_coeff(h, w):
	if h * w < 1e6:
		return 0.5
	s = max(max(h,w)/640, min(h,w)/480)
	return 0.5 * round(1/s, 3)

def inference_video(all_frames, DET, shot):
	bs = 32
	start_frame = shot[0].frame_num
	dets = []
	s = define_scale_coeff(*all_frames[0].shape[:-1])
	if len(all_frames) % bs == 0:
		batch_num = len(all_frames) // bs
	else:
		batch_num = len(all_frames) // bs + 1
	for batch_id in range(batch_num):
		batch_bboxes = DET.detect_faces(all_frames[batch_id*bs:(batch_id+1)*bs], conf_th=0.9, scales=[s])
		for batch_frame, bboxes in enumerate(batch_bboxes):
			frame_num = bs * batch_id + batch_frame + start_frame
			dets.append([])
			for bbox in bboxes:
				dets[-1].append(
					{
						'frame':frame_num, 
						'bbox':(bbox[:-1]).tolist(), 
						'conf':bbox[-1]}
					) 
	return dets

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = np.array([ f['frame'] for f in track ])
			bboxes      = np.array([np.array(f['bbox']) for f in track])
			frameI      = np.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = np.stack(bboxesI, axis=1)
			if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, all_frames, original_audio, shot):
	start_frame = shot[0].frame_num
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	face_all_frames = np.zeros((len(track["frame"]), 224, 224, 3), dtype="uint8")
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]  # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = all_frames[frame - start_frame]
		frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		face_all_frames[fidx] = cv2.resize(face, (224, 224))

	audioStart  = (track['frame'][0]) / FPS
	audioEnd    = (track['frame'][-1] + 1) / FPS
	audio = original_audio[int(audioStart*16000):int(audioEnd*16000)]
	save_dict = {
		'track': track, 
		'proc_track': dets
	}
	media_dict = {
		"audio": audio, 
		"video": face_all_frames
	}
	return save_dict, media_dict

def extract_MFCC(file, outPath):
	sr, audio = wavfile.read(file)
	mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
	featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
	np.save(featuresPath, mfcc)

def calculate_face_embeddings(facenet, faces):
	bs = 5000
	all_embeddings = np.zeros((len(faces), 512))
	inp_faces = np.stack([cv2.resize(elem, (160,160)) for elem in faces])
	inp_faces = np.einsum('bhwc->bchw', inp_faces)
	inp_faces = (inp_faces - 127.5)/128
	inp_faces = torch.from_numpy(inp_faces).float()
	for i in range(len(faces)//bs + 1):
		with torch.no_grad():
			embeddings = facenet(inp_faces[bs*i:bs*(i+1)].to(device))
			embeddings = embeddings.detach().cpu().numpy()
		all_embeddings[bs*i:bs*(i+1)] = embeddings
	return all_embeddings

def evaluate_network(s, media_list):
	allScores = []
	durationSet = {1,2,3,4,5,6} # Use this line can get more reliable result
	for item in media_list:
		audio = item["audio"]
		video = torch.from_numpy(item["video"])
		video = video[..., [2,1,0]]
		video_gs = torchvision.transforms.functional.rgb_to_grayscale(torch.permute(video, (0,3,1,2))).squeeze()
		videoFeature = video_gs[:, int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
		audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0] / FPS)
		audioFeature = audioFeature[:int(round(length * 100)),:]
		videoFeature = videoFeature[:int(round(length * FPS)),:,:]
		allScore = [] # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).to(device)
					inputV = videoFeature[int(i * duration * FPS): int((i+1) * duration * FPS),:,:].float().unsqueeze(0).to(device)
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)	
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels = None)
					scores.extend(score)
			allScore.append(scores)
		allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
		allScores.append(allScore)	
	return allScores

def visualization(tracks, scores, args):
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	faces = [[] for i in range(len(flist))]
	for tidx, track in enumerate(tracks):
		score = scores[tidx]
		for fidx, frame in enumerate(track['track']['frame'].tolist()):
			s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
			s = np.mean(s)
			faces[frame].append({'track':tidx, 'score':float(s),'s':track['proc_track']['s'][fidx], 'x':track['proc_track']['x'][fidx], 'y':track['proc_track']['y'][fidx]})
	firstImage = cv2.imread(flist[0])
	fw = firstImage.shape[1]
	fh = firstImage.shape[0]
	vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), FPS, (fw,fh))
	colorDict = {0: 0, 1: 255}
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		for face in faces[fidx]:
			clr = colorDict[int((face['score'] >= 0))]
			txt = round(face['score'], 1)
			cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,clr,255-clr),10)
			cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
		vOut.write(image)
	vOut.release()
	command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
		(os.path.join(args.pyaviPath, 'video_only.avi'), args.audioFilePath, \
		args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
	output = subprocess.call(command, shell=True, stdout=None)


# Main function
def main():
	args.pyaviPath = os.path.join(args.savePath, 'pyavi')
	args.pyframesPath = os.path.join(args.savePath, 'pyframes')
	args.pyworkPath = os.path.join(args.savePath, 'pywork')
	os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
	os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
	os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
	t = time()
	scenes = read_scenes(args)
	flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
	flist.sort()
	original_audio = read_audio(args)
	vidTracks = list()
	faces = list()
	all_scores = list()
	DET = S3FD(device=device)
	facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
	s = talkNet(device=device)
	s.loadParameters(args.pretrainModel)
	s.eval()
	print("Preprocessing time in s: ", time() - t)
	t = time()
	for shot in scenes:
		all_frames = read_video(flist, shot)
		shot_faces = inference_video(all_frames, DET, shot)
		faces.extend(list(shot_faces))
		if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
			tracks = track_shot(args, copy.deepcopy(shot_faces))
			if tracks:
				media_list = []
				for track in tracks:
					save_dict, media_dict = crop_video(args, track, all_frames, original_audio, shot)
					embeddings = calculate_face_embeddings(facenet, media_dict["video"])
					save_dict.update({"embeddings": embeddings})
					vidTracks.append(save_dict)
					media_list.append(media_dict)
				scores = evaluate_network(s, media_list)
				all_scores.extend(scores)
	print("Inference time in s: ", time() - t)
			
	savePath = os.path.join(args.pyworkPath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)
	fil = open(savePath, 'rb')
	vidTracks = pickle.load(fil)

	savePath = os.path.join(args.pyworkPath, 'scores.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(all_scores, fil)

	if args.visualisation:
		visualization(vidTracks, all_scores, args)	

if __name__ == '__main__':
    main()
