import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0])-np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle>180.0:
        angle=360-angle
    return angle

#define bow variables
counter=0
stage= None
text= None
#face detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# video feed
cap = cv2.VideoCapture(0)

#make detections
#50 % is good for tradeoffs
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        success, image2 = cap.read()
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image2 = cv2.cvtColor(cv2.flip(image2, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image2.flags.writeable = False

        # Get the result
        results = face_mesh.process(image2)

        # To improve performance
        image2.flags.writeable = True

        # Convert the color space from RGB to BGR
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image2.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                cv2.line(image2, p1, p2, (255, 0, 0), 2)


        # Detect stuff and render
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #recolor image
        image.flags.writeable = False

        results = pose.process(image) #make detection

        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) #recolor back to BGR (opencv wants image in BGR)


        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #print(landmarks) #print all landmarks xyz from 0 to 32 order
            #print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]) #print a specific landmark

            #Get coordintes
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]


            #calculate angle
            angle = calculate_angle(shoulder,hip,knee)
            #print(angle)
            cv2.putText(image,str(angle),
                        tuple(np.multiply(hip,[640, 480]).astype(int)),
                              cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255,255),2,cv2.LINE_AA
                        )
          #  print(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility)
            #Bow Counter Logic
            if angle > 165 and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility>0.3 or landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility>0.3):
                stage="Standing"
            if angle < 165 and stage =='Standing' and (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility>0.3 or landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility>0.3):
                stage="Bowing"
                counter+=1


        except:
            pass #if no detection just step out the loop

        #Render counter logic

        #setup box
       # cv2.rectangle(image, (0,0),(225,73),(245,117,16),-1)
        #print bow counter in box
        cv2.putText(image,'Count',(10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        #print Current position in box
        cv2.putText(image,'Posture',(10,105),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(image, stage, (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #Head direction print
        cv2.putText(image,'Head Direction',(10,205),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
        # Add the text on the image
        cv2.putText(image, text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,cv2.LINE_AA)
        #render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        #print(results.pose_landmarks) view xyz of each landmark
        #print(mp_pose.POSE_CONNECTIONS) view the connection between landmarks

        cv2.imshow('Mediapipe Feed',image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()





