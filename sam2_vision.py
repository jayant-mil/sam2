# First import the library
import pyrealsense2 as rs
import torch
import threading
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont 
import random
from sam2_vision.visualization import draw_polygons, draw_points
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Sam2Vision(threading.Thread):
    def __init__(self, pipeline, checkpoint, model_cfg):
        super().__init__()
        if torch.cuda.is_available():
            device = torch.device("cuda")

        if device == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16)

        self.pipeline = pipeline
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))
        self.positive_points = []
        self.negative_points = []
        self.interest_region_size = 50
        self.detected_objects = None
        self.init_image = None
        self.detecting = False
        self.initialize = False
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Masks', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('RealSense', self.mouse_callback)

    def run(self):
        try:
            with torch.inference_mode():
                while True:
                    # Create a pipeline object. This object configures the streaming camera and owns it's handle
                    frames = self.pipeline.wait_for_frames()
                    depth = frames.get_depth_frame()
                    color = frames.get_color_frame()
                    if not depth or not color: 
                        continue

                    color_image = np.asanyarray(color.get_data())
                    depth_image = np.asanyarray(depth.get_data())

                    if self.detecting:
                        if self.initialize:
                            self.init_image = color_image
                            self.predictor.set_image(color_image)
                            self.initialize = False
                        
                        # print(np.concatenate((self.positive_points, self.negative_points)))
                        # print(np.concatenate([[1]*len(self.positive_points),[0]*len(self.negative_points)]))
                        masks, _, _ = self.predictor.predict(np.concatenate([self.positive_points, self.negative_points]),
                                                            np.concatenate([[1]*len(self.positive_points),[0]*len(self.negative_points)]))
                        
                        print(f"{np.min(masks, axis=1)[1:]}, {np.max(masks, axis=1)[1:]}")
                        masked = cv2.bitwise_and(color_image, color_image, masks[0])
                        cv2.imshow('Masks', masked)
                    # Draw a point of interest
                    for point in self.positive_points:
                        cv2.circle(color_image, point, 5, (0, 255, 0), -1)

                    for point in self.negative_points:
                        cv2.circle(color_image, point, 5, (255, 0, 0), -1)

                    cv2.imshow('RealSense', color_image)
                    cv2.waitKey(1)
                    
                    # Get click event and position
                

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.override_existing_point((x, y))
            if flags == cv2.EVENT_FLAG_SHIFTKEY:
                self.negative_points.append((x, y))
            else:
                self.positive_points.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:
            if flags == cv2.EVENT_FLAG_SHIFTKEY:
                self.positive_points = []
                self.negative_points = []
            else:
                self.initialize = True
                self.detecting = True
    
    def override_existing_point(self, point):
        for i, p in enumerate(self.positive_points):
            if np.linalg.norm(np.array(p) - np.array(point)) < self.interest_region_size:
                self.positive_points.pop(i)
        for i, p in enumerate(self.negative_points):
            if np.linalg.norm(np.array(p) - np.array(point)) < self.interest_region_size:
                self.negative_points.pop(i)

                
if __name__ == '__main__':
    pipeline = rs.pipeline()
    rsconfig = rs.config()

    rsconfig.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
    rsconfig.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(rsconfig)

    sam2_checkpoint = "/sam_checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    vision = Sam2Vision(pipeline, sam2_checkpoint, model_cfg)
    vision.start()
    vision.join()

