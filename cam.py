import cv2

cap = cv2.VideoCapture(0)
#scale_factor = 0.75

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame.")
        break
        
    #frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
    
    cv2.imshow("Original", frame)

    # Salida con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()