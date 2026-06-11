import mediapipe as mp 
import cv2 
import time 
import math


class PoseTracking():

           def __init__(self,num_poses = 1,min_pose_detection_confidence = 0.6,min_pose_presence_confidence = 0.6,min_tracking_confidence = 0.6):
           
                  self.num_poses = num_poses
                  self.min_pose_detection_confidence = min_pose_detection_confidence
                  self.min_pose_presence_confidence = min_pose_presence_confidence
                  self.min_tracking_confidence = min_tracking_confidence
                  model_path = "pose_landmarker_heavy.task"
                  
                  option = mp.tasks.vision.PoseLandmarkerOptions(
                   base_options = mp.tasks.BaseOptions(model_asset_path = model_path),
                   running_mode = mp.tasks.vision.RunningMode.VIDEO,
                   num_poses = self.num_poses,
                   min_pose_detection_confidence = self.min_pose_detection_confidence,
                   min_pose_presence_confidence = self.min_pose_presence_confidence,
                   min_tracking_confidence = self.min_tracking_confidence
                   )
                   
                  self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(option)
                  self.ctime = 0
                  self.ptime = 0
                  
                  self.POSE_CONNECTIONS = [
                     (11, 13), (13, 15),   # Left arm
                    (12, 14), (14, 16),   # Right arm
                    (11, 12),             # Shoulders
                    (11, 23), (12, 24),   # Torso
                    (23, 24),             # Hips
                    (23, 25), (25, 27),   # Left leg
                    (24, 26), (26, 28)    # Right leg
                  ]
                  
           def Findpose(self,img,draw=False,drawLine=False):
                  self.image = mp.Image(image_format=mp.ImageFormat.SRGB,data=img)
                  timestamp = int(time.time()*1000)
                  self.result = self.landmarker.detect_for_video(self.image,timestamp)
                  self.lmlist = []
                  if self.result.pose_landmarks:
                      for poseLms in self.result.pose_landmarks:
                         h,w,c = img.shape
                         
                         for id,lm in enumerate(poseLms):
                           cx,cy = int(lm.x*w),int(lm.y*h)
                           self.lmlist.append([id,cx,cy])
                           
                           if draw:
                             cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)
                         
                         if drawLine:
                            for st,end in self.POSE_CONNECTIONS:
                               x1, y1 = int(poseLms[st].x * w), int(poseLms[st].y * h)
                               x2, y2 = int(poseLms[end].x * w), int(poseLms[end].y * h)
                               cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)    
                         
                  return img,self.lmlist
                  
                  
           def FindAngel(self,img,p1,p2,p3,draw=True,position = (50,70)):
                
                x1,y1 = self.lmlist[p1][1:]
                x2,y2 = self.lmlist[p2][1:]
                x3,y3 = self.lmlist[p3][1:]

                angel = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))

                if angel < 0:
                     angel += 360
                if angel > 360:
                     angel = 360 - angel
                
                cv2.putText(img,f"Angle:{int(angel)}",position,cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
                if draw:
                     cv2.circle(img,(x1,y1),3,(255,0,255),3)
                     cv2.circle(img,(x2,y2),3,(255,0,255),3)
                     cv2.circle(img,(x3,y3),3,(255,0,255),3)

                return img,angel
                

           def FPS(self,img):
               self.cTime = time.time()
               fps = 1/(self.cTime - self.pTime)
               self.pTime = self.cTime
               cv2.putText(img,f"FPS:{int(fps)}",(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
               return img
   


def main():

    cap = cv2.VideoCapture(0)

    while True:
         _,img = cap.read()
         img = cv2.flip(img,1)
         cv2.imshow("camera",img)
         key = cv2.waitKey(10)
         if key == 27:
            break
    cap.release()
    cv2.destroyAllWindow()


if __name__ == "__main__":
     
      main()
