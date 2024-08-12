import cv2

# load the pre-trained face detection cascade classifier
face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

cap =cv2.VideoCapture(0)
while True:
    # capture frame by frame
    ret,frame =cap.read()
    # convert the frame to grayscale
    gray = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)

    # DETECT FACE IN THE FRAME
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1 , minNeighbors=5 ,minSize=(30,30))

    # draw rectangle around the faces
    for(x,y,w,h) in face :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255 ,0 ,0) , 2 )

    #  display the resulting frame
    cv2.imshow("face detection",frame)

    # break the loop when "q" is pressed
    if cv2.waitKey(1) & 0XFF ==ord("q") :
        break
# release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()