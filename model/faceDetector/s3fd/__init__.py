import time, os, sys, subprocess
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_

PATH_WEIGHT = 'model/faceDetector/s3fd/sfd_face.pth'
if os.path.isfile(PATH_WEIGHT) == False:
    Link = "1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt"
    cmd = "gdown --id %s -O %s"%(Link, PATH_WEIGHT)
    subprocess.call(cmd, shell=True, stdout=None)
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():
    def __init__(self, device):
        self.device = device
        self.net = S3FDNet(device=self.device).to(self.device)
        PATH = os.path.join(os.getcwd(), PATH_WEIGHT)
        state_dict = torch.load(PATH, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        _, h, w, _ = image.shape
        image = torch.from_numpy(image)
        s = scales[0] 
        scaled_img = np.zeros((len(image), int(h*s), int(w*s), 3))
        for i, frame in enumerate(image):
            scaled_img[i] = cv2.resize(frame.numpy(), dsize=(int(w*s), int(h*s)), interpolation=cv2.INTER_LINEAR)
            # scaled_img[i] = cv2.resize(frame.numpy(), dsize=(0,0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        scaled_img = torch.from_numpy(scaled_img)
        scaled_img = torch.permute(scaled_img, (0, 3, 1, 2)).float()
        all_bboxes = list()
        with torch.no_grad():
            scaled_img = scaled_img - img_mean[None, ...]
            scaled_img = scaled_img[:, [2, 1, 0], :]
            y_all = self.net(scaled_img.to(self.device))
            for y in y_all:
                bboxes = np.empty(shape=(0, 5))
                detections = y.data
                scale = torch.Tensor([w, h, w, h])
                for i in range(detections.size(0)):
                    j = 0
                    while detections[i, j, 0] > conf_th:
                        score = detections[i, j, 0]
                        pt = (detections[i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1
                keep = nms_(bboxes, 0.1)
                bboxes = bboxes[keep]
                if bboxes.size > 0:
                    all_bboxes.append(bboxes)
        return all_bboxes
