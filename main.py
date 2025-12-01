import cv2 as cv
import mediapipe as mp
import numpy as np
import math

# --- 1. Função de Cálculo de Ângulo (Simplificada) ---
def calculate_angle(a, b, c):
    # a, b, c são as coordenadas (x, y) dos landmarks
    a = np.array(a)
    b = np.array(b)  # Vértice
    c = np.array(c)
    
    # Cria vetores BA e BC
    ba = a - b
    bc = c - b
    
    # Produto escalar e norma para o cálculo do cosseno
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Calcula o ângulo em radianos e converte para graus
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    
    return angle

# --- 2. Inicialização do MediaPipe e OpenCV ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Usa a webcam (índice 0, geralmente)
cap = cv.VideoCapture(0) 

# Instância do Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

print("Iniciando detecção. Pressione 'q' para sair.")

# --- 3. Loop Principal ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler o frame da câmera.")
        break
    
    # Inverter a imagem horizontalmente (opcional, para espelhar a webcam)
    frame = cv.flip(frame, 1)

    # 1. Converter para RGB (necessário para o MediaPipe)
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False # Otimização de performance
    
    # 2. Processar a imagem
    results = pose.process(image)
    
    # 3. Desenhar os landmarks e calcular o ângulo
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # Voltar para BGR para exibição do OpenCV

    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Exemplo: Calcular o ângulo do cotovelo direito para um Bicep Curl
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calcular e formatar o ângulo
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Desenhar o ângulo na tela (Exemplo: 170.5°)
            cv.putText(image, f'Angulo: {angle:.1f}', 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), # Ajuste de coordenadas
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
            
            # Desenhar as conexões da pose
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            
    except Exception as e:
        # Erro se o landmark não for detectado no frame, o que é comum
        # print(f"Erro de processamento: {e}")
        pass
    
    # 4. Mostrar o resultado
    cv.imshow('MediaPipe Pose Feed', image)

    # Condição para sair: Pressione 'q'
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

# --- 4. Finalização ---
cap.release()
cv.destroyAllWindows()