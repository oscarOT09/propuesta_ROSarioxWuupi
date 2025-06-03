# Importaciones necesarias
import numpy as np
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Conexión con Unity
import socket
import select

# Manejo de archivos CSV
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Crear socket UDP
sock_botones = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address_botones = ('localhost', 5005)

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_address = ('localhost', 5053)
MAX_DGRAM = 65000

sock_log = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_log.bind(('localhost', 5060))
sock_log.setblocking(False)

user_receive = False
log_out = False
            
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

def send_button_selection(index, wrist):
    message = f"{index},{wrist}"
    sock_botones.sendto(message.encode(), server_address_botones)

def calcular_angulo(A, B, C):
    # Convertir puntos a vectores
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])

    # Normalizar vectores
    BA_norm = BA / np.linalg.norm(BA)
    BC_norm = BC / np.linalg.norm(BC)

    # Producto punto y ángulo
    cos_angle = np.dot(BA_norm, BC_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Evitar errores numéricos
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

while not user_receive:
    ready = select.select([sock_log], [], [], 0.1)  # Espera hasta 100ms
    if ready[0]:
      data, _ = sock_log.recvfrom(1024)
      message = data.decode()
      #print(f"[Socket Control] Mensaje recibido: {message}")
      if "user" in message.lower():
          user_receive = True
          user_name = message.split(":")[-1].strip()
      else:
          user_receive = True
          user_name = message.strip()
      print(f"[Socket Control] Activando procesamiento para el usuario: {user_name}")
    
####### Archivo CSV
csv_filename = f"Assets/PythonScripts/{user_name}_registro_angulos.csv"

# Crear encabezado si el archivo no existe
try:
    with open(csv_filename, 'x', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "lado", "angulo"])
except FileExistsError:
    pass

# Create an PoseLandmarker object
base_options = python.BaseOptions(model_asset_path='./Assets/PythonScripts/pose_landmarker_full.task')
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

write_interval = 1.0  # segundos
last_write_time = time.time()


while True:
  ret, frame = cap.read()

  if not ret:
    print("No se pudo leer el frame.")
    break
  
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
    left_ang = calcular_angulo(left_shoulder, left_elbow, left_wrist)
    right_ang = calcular_angulo(right_shoulder, right_elbow, right_wrist)


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
              send_button_selection(area_index + 1, wrist)
      else:
          current_areas[idx] = None
          entry_times[idx] = None

    ###### Escritura en el CSV
    current_time = time.time()
    if (current_time - last_write_time) >= write_interval:
        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([now, "izquierda", f"{left_ang:.2f}"])
            writer.writerow([now, "derecha", f"{right_ang:.2f}"])
        last_write_time = current_time
        print(f"Escritura existosa: {now}")

    # Impresión de la distancia en la imagen
    cv2.putText(frame, f'{int(left_ang)} degs',
                get_text_point(left_shoulder, left_wrist), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(frame, f'{int(right_ang)} degs',
                get_text_point(right_shoulder, right_wrist), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Vidsualización del frame procesado    
    frame = cv2.resize(frame, (320, 240))
    _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    data = img_encoded.tobytes()
    # Envio del frame 
    for i in range(0, len(data), MAX_DGRAM):
      chunk = data[i:i+MAX_DGRAM]
      sock_video.sendto(chunk, video_address)

    cv2.waitKey(1)

    try:
      data, addr = sock_log.recvfrom(1024)
      message = data.decode("utf-8").strip()
      if message.lower() == "true":
          print("Cierre de sesión solicitado.")
          break
    except BlockingIOError:
        # No hay datos disponibles; continuar con el ciclo normal
        pass

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
sock_botones.close()
sock_log.close()
sock_video.close()

### Graficación
# Cargar datos con pandas
df = pd.read_csv(
    csv_filename,
    parse_dates=["timestamp"],
    date_parser=lambda x: pd.to_datetime(x, format="%d/%m/%Y %H:%M:%S")
)

df["angulo"] = pd.to_numeric(df["angulo"], errors='coerce')

hoy = datetime.now().date()
df_hoy = df[df["timestamp"].dt.date == hoy]

df_hoy["angulo"] = pd.to_numeric(df_hoy["angulo"], errors='coerce')

plt.figure(figsize=(12, 5))
for lado in ["izquierda", "derecha"]:
    datos = df_hoy[df_hoy["lado"] == lado]
    plt.plot(datos["timestamp"], datos["angulo"], label=f"Codo {lado}")
plt.title("Progreso de extensión del codo hoy")
plt.xlabel("Hora")
plt.ylabel("Ángulo (°)")
plt.legend()
plt.xticks(rotation=45)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

plt.tight_layout()
plt.grid()

# --- Promedio últimos 7 días ---
fecha_inicio = datetime.now().date() - timedelta(days=6)
df["fecha"] = df["timestamp"].dt.date
df_7dias = df[df["fecha"] >= fecha_inicio]

promedios = df_7dias.groupby(["fecha", "lado"])["angulo"].mean().unstack()

#plt.figure(figsize=(10, 5))
promedios.plot(marker='o')
plt.title("Promedio diario de extensión del codo (últimos 7 días)")
plt.xlabel("Fecha")
plt.ylabel("Ángulo promedio (°)")
plt.grid()
plt.tight_layout()
plt.show()