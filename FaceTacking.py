import mediapipe as mp
import time
import cv2


class FaceTracking:

    def __init__(self,num_faces=1,min_face_detection_confidence=0.5,
                 min_face_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 output_face_blendshapes=False,
                 output_facial_transformation_matrixes=False):
        
        self.num_faces = num_faces
        self.min_face_detection_confidence = min_face_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_face_presence_confidence = min_face_presence_confidence
        self.output_face_blendshapes = output_face_blendshapes
        self.output_facial_transformation_matrixes = output_facial_transformation_matrixes
        model_path = "face_landmarker.task"

        option = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = mp.tasks.BaseOptions(model_asset_path = model_path),
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            num_faces = self.num_faces,
            min_face_detection_confidence = self.min_face_detection_confidence,
            min_face_presence_confidence = self.min_face_presence_confidence,
            min_tracking_confidence = self.min_tracking_confidence,
            output_face_blendshapes = self.output_face_blendshapes,
            output_facial_transformation_matrixes = self.output_facial_transformation_matrixes
        )

        self.facelandmark = mp.tasks.vision.FaceLandmarker.create_from_options(option)
        self.ctime = 0
        self.ptime = 0


    def FindFace(self,img,draw=True):

        self.lmlist = []
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=img)
        timestamp = int(time.time()*1000)
        result = self.facelandmark.detect_for_video(mp_image,timestamp)
        if result.face_landmarks:
            for facelms in result.face_landmarks:
                for id,lms in enumerate(facelms):
                    h,w,_ = img.shape
                    cx,cy = int(lms.x*w), int(lms.y*h)
                    self.lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),1,(0,255,0),1)

        return img,self.lmlist



    def FPS(self,img):
        self.ctime = time.time()
        fps = 1/(self.ctime-self.ptime)
        self.ptime = self.ctime
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
