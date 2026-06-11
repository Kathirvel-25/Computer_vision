import mediapipe as mp 
import cv2
import time
import math


class Hand_Tracking:

    def __init__(self,num_hands = 1, min_hand_detection_confidence = 0.5,
                 min_hand_presence_confidence = 0.5,
                 min_tracking_confidence = 0.5):
        
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        model_path = "hand_landmarker.task"
        self.Hand_connection = [
                 (0,1),(1,2),(2,3),(3,4),
                 (0,5),(5,6),(6,7),(7,8),
                 (5,9),(9,10),(10,11),(11,12),
                 (9,13),(13,14),(14,15),(15,16),
                  (13,17),(17,18),(18,19),(19,20)    
                 ]
        option = mp.tasks.vision.HandLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path = model_path),
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            num_hands = self.num_hands,
            min_hand_detection_confidence = self.min_hand_detection_confidence,
            min_hand_presence_confidence = self.min_hand_presence_confidence,
            min_tracking_confidence = self.min_tracking_confidence
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(option)
        self.cTime = 0
        self.pTime = 0
        
        
    def FindHand(self,img,draw=True,drawLine = True): 

         self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=img)
         timestamp = int(time.time() * 1000)
         self.result = self.landmarker.detect_for_video(self.mp_image,timestamp)
         self.lmlist = []
         if self.result.hand_landmarks:
             for handlms in self.result.hand_landmarks:
                 for id,lm in enumerate(handlms):
                      h,w,c = img.shape
                      cx,cy = int(lm.x*w), int(lm.y*h)
                      self.lmlist.append([id,cx,cy])
                      if draw:
                           cv2.circle(img,(cx,cy),6,(0,255,255),cv2.FILLED)
                      if drawLine:     
                       for st,end in self.Hand_connection:
                        x1,y1 = int(handlms[st].x*w), int(handlms[st].y*h)
                        x2,y2 = int(handlms[end].x*w), int(handlms[end].y*h)
                        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3)
         return img,self.lmlist

       
    def FindDistance(self,img,p1,p2,draw=True): # This to Find the Distance bet the two point 

        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        cx = (x1 + x2) // 2
        cy = (y1+y2) // 2
        distance = int(math.hypot(x2-x1,y2-y1))
        if draw:
            cv2.circle(img,(x1,y1),8,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),8,(255,0,255),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)
            cv2.circle(img,(cx,cy),8,(255,0,255),cv2.FILLED)
        return img,distance

 
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
          cv2.imshow("Camera",img)
          key = cv2.waitKey(10)
          if key == 27:
              break
     cap.release()
     cv2.destroyAllWindows()    



if __name__ == "__main__":
    main()
