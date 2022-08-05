import cv2
import imutils as imutils
import numpy as np
from matplotlib import pyplot as plt



class videoprocess:
    def __init__(self, fileName):
        self.fileName=fileName

    def mask (self, frame) :
        image = frame
        original = image.copy()
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #lower = np.array([0, 0, 0])   #0,18,0
        #upper = np.array([255, 255, 255])# 88,255,139
        #masked= cv2.inRange(image,lower,upper)
        (thresh, blackAndWhiteImage)= cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

        mask_2=cv2.adaptiveThreshold(blackAndWhiteImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 199, 5)

        return grayImage ;

    def find_largest_contour(self, image_2):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        original_image = image_2
        original_image = imutils.resize(original_image,
                               width=min(400, original_image.shape[1]))
        (regions, _) = hog.detectMultiScale(original_image,
                                            winStride=(4, 4),
                                            padding=(4, 4),
                                            scale=1.05)

        # Drawing the regions in the Image
        for (x, y, w, h) in regions:
            cv2.rectangle(original_image, (x, y),
                          (x + w, y + h),
                          (0, 0, 255), 5)

           # print (x,y,w,h)
       # cv2.imshow('Detection',original_image)
        #cv2.waitKey(3000)
        return original_image
    def isolateforeground(self,frame):
        new_img= self.mask(frame)
        self.find_largest_contour(new_img)


    def videoframeCapture(self):
        points_collect= []
        key=cv2.waitKey(20)
        vid_record = cv2.VideoCapture(self.fileName) ;
        length = int(vid_record.get(cv2.CAP_PROP_FRAME_COUNT))
        print(length)
        while (vid_record.isOpened()):
            ret = vid_record.grab();
            counter =3
            target= int(length/(8))
            print(target)
            while ret:
                vid_record.set(1, counter)
                ret,frame=vid_record.read()
                cv2.waitkey(1000)
                new_frame=self.find_largest_contour(frame);
            
                frames, point= self.poseDetector(new_frame)
                cv2.waitkey(1000)
                points_collect.extend(point)

                counter+=target
                if (counter+target)>= length :
                    vid_record.release()
                    cv2.destroyAllWindows()
                    break
                print(5)
                if key == ord('q'):
                    break
        print(6)

        print(points_collect)





    def poseDetector(self,frame):
        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

        width = 299
        height = 299
        inWidth = width
        inHeight = height

        net = cv2.dnn.readNetFromTensorflow('C:/Users/vijay_000/Documents/Youcam/graph_opt.pb')
        thr = 0.2
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert (len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            #plt.plot(x, y, color='green', linestyle='dashed', linewidth=3,
                    # marker='o', markerfacecolor='blue', markersize=12)
            points.append((int(x), int(y)) if conf > thr else None)
      #  plt.ylim(1, 1000)
       # plt.xlim(1, 1000)
       # plt.xlabel('x - axis')
       # plt.ylabel('y - axis')
       # plt.title('Pose graph')
       # plt.show()
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert (partFrom in BODY_PARTS)
            assert (partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = net.getPerfProfile()

        return frame, points






