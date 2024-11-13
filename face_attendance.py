import face_recognition
import numpy as np
import cv2
import csv
import datetime

video_capture = cv2.VideoCapture(0) #Captures the video as an input

first_person=face_recognition.load_image_file("")
first_person_encodings = face_recognition.face_encodings(first_person)

second_person= face_recognition.load_image_file("")
second_person_encodings = face_recognition.face_encodings(second_person)

known_face_encodings = [first_person_encodings[0], second_person_encodings[0]]
known_face_names = [""] #Enter the names of the people of which data were given

attend=known_face_names.copy()

now=datetime.datetime.now()
t=now.strftime("%Y-%m-%d- %I_%M_%S_%p")
f=open(f"{t}.csv", "w+", newline="")
lnwriter = csv.writer(f)

#Running an infinite loop until a particular condition which is kept at last
while True:
    _,frame=video_capture.read()
    rgb_actual_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if True:
        face_locations=face_recognition.face_locations(rgb_actual_frame)
        face_encodings=face_recognition.face_encodings(rgb_actual_frame, face_locations)

        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
            face_distances=face_recognition.face_distance(known_face_encodings,face_encoding)
            best_match_index=np.argmin(face_distances)

            if matches[best_match_index]:
                name=known_face_names[best_match_index]

                if name in known_face_names:
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    bottom=(10,100)
                    fontScale=0.5
                    fontcolor=(255,255,255)
                    thickness=2
                    linetype=2
                    cv2.putText(frame,name,bottom,font,fontScale,fontcolor,linetype)

                    if name in attend:
                        now=datetime.datetime.now()
                        t=now.strftime("%I_%M_%S_%p")
                        attend.remove(name)
                        lnwriter.writerow([name, t])



    cv2.imshow("Attendance System",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()