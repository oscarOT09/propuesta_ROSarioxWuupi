from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


'''def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image'''

def get_point(art_index, w, h):
  x_cord = int(detection_result.pose_landmarks[0][art_index].x * w)
  y_cord = int(detection_result.pose_landmarks[0][art_index].y * h)
  return x_cord, y_cord

def get_distance(start, end):
  text_x = int(start[0] + (end[0] - start[0]) / 2)
  text_y = int(start[1] + (end[1] - start[1]) / 2)

  distance = np.linalg.norm(np.array(start) - np.array(end))

  return distance, text_x, text_y

# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
                                      base_options=base_options,
                                      output_segmentation_masks=True
                                      )
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
#scale_factor = 0.75
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

indices = [11, 12, 13, 14, 15, 16] # left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist
colores = [(255,0,0), (255,0,0), (0,255,0), (0,255,0), (0,0,255), (0,0,255)] 
circulos_arts = []

while True:
  ret, frame = cap.read()
  if not ret:
      print("No se pudo leer el frame.")
      break
      
  #frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
  detection_result = detector.detect(mp_image)
  
  if len(detection_result.pose_landmarks) > 0:
    h, w, _ = frame.shape
    
    idx0 = 0
    for idx in indices:
      x_cord = int(detection_result.pose_landmarks[0][idx].x * w)
      y_cord = int(detection_result.pose_landmarks[0][idx].y * h)
      
      circulos_arts.append(np.array([x_cord, y_cord]))
      cv2.circle(frame, (x_cord, y_cord), 5, colores[idx0], -1)

      cv2.putText(frame,  
              f'({x_cord}, {y_cord})',  
              (x_cord-5, y_cord-5),  
              cv2.FONT_HERSHEY_SIMPLEX, 0.5,  
              (0, 255, 255),  
              1,  
              cv2.LINE_4)
      
      idx0 += 1

    # Articulaciones izquierdas
    left_shoulder = get_point(11, w, h)
    left_elbow    = get_point(13, w, h)
    left_wrist    = get_point(15, w, h)

    # Articulaciones derechas
    right_shoulder = get_point(12, w, h)
    right_elbow    = get_point(14, w, h)
    right_wrist    = get_point(16, w, h)

    # Lineas entre articulaciones
    cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 1)
    cv2.line(frame, left_elbow, left_wrist, (0, 0, 255), 1)
    cv2.line(frame, left_shoulder, left_wrist, (255, 0, 0), 1)
    cv2.line(frame, right_shoulder, right_elbow, (0, 255, 0), 1)
    cv2.line(frame, right_elbow, right_wrist, (0, 0, 255), 1)
    cv2.line(frame, right_shoulder, right_wrist, (255, 0, 0), 1)

    # Cálculo de la distancia euclidiana entre hombros y muñecas
    left_distance, left_text_x, left_text_y = get_distance(left_shoulder, left_wrist)
    right_distance, right_text_x, right_text_y = get_distance(right_shoulder, right_wrist)

    # Impresión de la distancia en la imagen
    cv2.putText(frame, f'Dist: {int(left_distance)} px',
                (left_text_x, left_text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f'Dist: {int(left_distance)} px',
                (right_text_x, right_text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Vidsualización del frame procesado
    cv2.imshow("Seleccion de botones | TEC PUE x Wuupi", frame)
    
    # Salida con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()