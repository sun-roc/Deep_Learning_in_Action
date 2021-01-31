import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlehub as hub
import cv2
import numpy as np

cap = cv2.VideoCapture("2.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter('2.avi',cv2.VideoWriter_fourcc(*'XVID'),20,size)
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
object_detector_res = hub.Module(name="pyramidbox_lite_server_mask")
i = 0
while(1):
    ret, frame = cap.read()
    i = i+1
    if ret==True:
        list_result = object_detector_res.face_detection(images=[frame],use_gpu=True,output_dir='detection_result',visualization= True)
        print(list_result[0]["path"][:-2]+".jpg")
        out_feature = cv2.imread("./detection_result/"+list_result[0]["path"][:-2]+".jpg")
        cv2.imshow("mat",out_feature)
        # out_feature = cv2.putText(out_feature, "{}".format(len(list_result[0]["data"])), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        out.write(out_feature)

        print(len(list_result[0]["data"]),list_result[0]["data"])
        # cv2.imshow('frame',out_feature)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows() 