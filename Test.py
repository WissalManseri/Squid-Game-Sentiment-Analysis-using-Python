from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/haarcascade_frontalface_default.xml')
classifier =load_model('/Users/durgeshthakur/Deep Learning Stuff/Emotion Classification/Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

# def face_detector(img):
#     # Convert image to grayscale
#     grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray,1.3,5)
#     if faces is ():
#         return (0,0,0,0),np.zeros((50,50),np.uint8),img

#     for (x,y,w,h) in faces:
#         cv2.dw.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h,x:x+w,h:x+y]

     try:
         roi_gray = cv2.dw.resize(roi_gray,(50,50),interpolation=cv2.INTER_AREA)
     except: 
         return (x,w,y,h),np.zeros((50,50),np.uint8,roi_grey),img
     return (x,w,y,h),roi_gray,img
     try:
     k = 5//0 # raises divide by zero exception.
     print(k)
    
# handles zerodivision exception    
except ZeroDivisionError:   
    print("Can't divide by zero")

#take input 
cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROIT, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
ender_mail = 'sender@fromdomain.com'    
receivers_mail = ['reciever@todomain.com']    
message = """From: From Person %s  
To: To Person %s  
Subject: Sending SMTP e-mail   
This is a test e-mail message.  
"""%(sender_mail,receivers_mail)    
try:    
   smtpObj = smtplib.SMTP('localhost')    
   smtpObj.sendmail(sender_mail, receivers_mail, message)    
   print("Successfully sent email")    
except Exception:    
   print("Error: unable to send email")    
#Exit
cap.release()
cv2.destroyAllWindows()



























