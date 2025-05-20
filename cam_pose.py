import numpy as np
import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def get_point(art_index, w, h):
  x_cord = int(detection_result.pose_landmarks[0][art_index].x * w)
  y_cord = int(detection_result.pose_landmarks[0][art_index].y * h)
  return x_cord, y_cord

def get_text_point(start, end):
  text_x = int(start[0] + (end[0] - start[0]) / 2)
  text_y = int(start[1] + (end[1] - start[1]) / 2)

  return text_x, text_y

def get_area_index(x, y, areas):
    for i, ((x1, y1), (x2, y2)) in enumerate(areas):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

# Create an PoseLandmarker object
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
                                      base_options=base_options,
                                      output_segmentation_masks=True
                                      )
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

indices = [11, 12, 13, 14, 15, 16] # left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist
arts_colors = [(255,0,0), (255,0,0), (0,255,0), (0,255,0), (0,0,255), (0,0,255)] 
button_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

current_areas = [None, None]     # Uno para cada muñeca
entry_times = [None, None]

new_width = 176
new_height = 120

while True:
  ret, frame = cap.read()
  if not ret:
      print("No se pudo leer el frame.")
      break
  #frame = cv2.resize(frame, (new_width, new_height))
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  frame_height, frame_width, _ = frame.shape

  area_height = int(frame_height * 0.25)
  area_width = int(frame_width / 4)

  # Definir coordenadas de las 4 áreas (x1, y1, x2, y2)
  areas = [
          ((0, 0), (area_width, area_height)),                    # Área 0
          ((area_width, 0), (2 * area_width, area_height)),       # Área 1
          ((2 * area_width, 0), (3 * area_width, area_height)),   # Área 2
          ((3 * area_width, 0), (frame_width, area_height))       # Área 3
          ]

  overlay = frame.copy()  # Capa que tendrá los rectángulos transparentes
  for i, ((x1, y1), (x2, y2)) in enumerate(areas):
      cv2.rectangle(overlay, (x1, y1), (x2, y2), button_colors[i], -1)  # Rellenar en overlay
      cv2.putText(overlay, f'Boton {i+1}', (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
  alpha = 0.25  # 50% de transparencia
  cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # Mezcla final en 'frame'

  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
  detection_result = detector.detect(mp_image)
  
  if len(detection_result.pose_landmarks) > 0:    
    idx0 = 0

    for idx in indices:
      tmp_cord = get_point(idx, frame_width, frame_height)

      cv2.circle(frame, tmp_cord, 5, arts_colors[idx0], -1)
      cv2.putText(frame,  
              f'({tmp_cord[0]}, {tmp_cord[1]})',  
              (tmp_cord[0]-5, tmp_cord[1]-5),  
              cv2.FONT_HERSHEY_SIMPLEX, 0.5,  
              (0, 255, 255),  
              1,  
              cv2.LINE_4)
      
      idx0 += 1

    # Articulaciones izquierdas
    left_shoulder = get_point(11, frame_width, frame_height)
    left_elbow    = get_point(13, frame_width, frame_height)
    left_wrist    = get_point(15, frame_width, frame_height)

    # Articulaciones derechas
    right_shoulder = get_point(12, frame_width, frame_height)
    right_elbow    = get_point(14, frame_width, frame_height)
    right_wrist    = get_point(16, frame_width, frame_height)

    # Lineas entre articulaciones
    cv2.line(frame, left_shoulder, left_elbow, (0, 255, 0), 1)
    cv2.line(frame, left_elbow, left_wrist, (0, 0, 255), 1)
    cv2.line(frame, left_shoulder, left_wrist, (255, 0, 0), 1)
    cv2.line(frame, right_shoulder, right_elbow, (0, 255, 0), 1)
    cv2.line(frame, right_elbow, right_wrist, (0, 0, 255), 1)
    cv2.line(frame, right_shoulder, right_wrist, (255, 0, 0), 1)

    # Cálculo del ángulo entre hombros y muñecas
    dx_left = abs(left_wrist[0] - left_shoulder[0])
    dy_left = abs(left_wrist[1] - left_shoulder[1])

    left_ang = np.rad2deg(np.arctan2(dy_left, dx_left))

    dx_right = abs(right_wrist[0] - right_shoulder[0])
    dy_right = abs(right_wrist[1] - right_shoulder[1])

    right_ang = np.rad2deg(np.arctan2(dy_right, dx_right))

    arts_array = [left_wrist, right_wrist]

    # Comprobación de detección de botón
    for idx, art in enumerate(arts_array):
      area_index = get_area_index(art[0], art[1], areas)
      if area_index is not None:
          if area_index != current_areas[idx]:
              current_areas[idx] = area_index
              entry_times[idx] = time.time()
          else:
              elapsed_time = time.time() - entry_times[idx]
              if elapsed_time >= 2.0:
                  if idx == 0:
                     wrist = "izquierda"
                  else:
                     wrist = "derecha"
                  
                  print(f'Botón virtual {area_index + 1} PRESIONADO con muñeca {wrist}')
                  # Aquí podrías añadir flags de activación si no quieres múltiples prints
      else:
          current_areas[idx] = None
          entry_times[idx] = None


    # Impresión de la distancia en la imagen
    cv2.putText(frame, f'{int(left_ang)} degs',
                get_text_point(left_shoulder, left_wrist), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f'{int(right_ang)} degs',
                get_text_point(right_shoulder, right_wrist), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Vidsualización del frame procesado
    cv2.imshow("Seleccion de botones | TEC PUE x Wuupi", frame)
    
    # Salida con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()