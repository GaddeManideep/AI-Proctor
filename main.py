import cv2
import mediapipe as mp
import numpy as np
import threading as th
import sounddevice as sd
import soundfile as sf
from deepface import DeepFace
import time
import pyaudio
import wave
import os
import keyboard
import speech_recognition as sr
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# Global variables
x = 0  # X axis head pose
y = 0  # Y axis head pose

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0
flag_event = th.Event()
ALARM_SOUND = "alarm.wav"  # Path to your alarm sound file
net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
input_size = (640, 480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

reference_img = cv2.imread("ref.jpg")

check = False
cheat = False
cellphone_cheat = False
warnings = 0
NO_MATCH_THRESHOLD=10

# Global variables to track time spent in each direction
time_left = 0
time_right = 0
time_down = 0

def check_face(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    num_faces = len(faces)
    face_match = False

    try:
        if DeepFace.verify(image, reference_img.copy())['verified']:
            face_match = True
    except ValueError:
        pass

    return face_match, num_faces

def pose():
    global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT, check, cheat, cellphone_cheat, time_left, time_right, time_down, time_threshold,warnings,checkcount,no_match_time
    check = False
    cheat = False
    cellphone_cheat = False
    time_left = 0
    time_right = 0
    time_down = 0
    time_threshold = 3  # Threshold time in seconds for triggering the alarm
    neutral_threshold = 5  # Threshold for considering the head pose as neutral
    neutral_count = 0
    warnings = 0 
    checkcount=0
    no_match_time = 0 
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        face_match, num_faces = check_face(image)
        

        # Resize the image and normalize it
        if not face_match:
            no_match_time += 1  # Increment no match time if there's no match
        else:
            no_match_time = 0  # Reset no match time if there's a match
        
        # Check if no match time exceeds threshold and release capture if it does
        if no_match_time > NO_MATCH_THRESHOLD:
            flag = True
            flag_event.set()
            print("rec stoppped due to fach doesnt match")
            cap.release()
            break
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, input_size, swapRB=True, crop=False)
        net.setInput(blob)

        # Perform forward pass
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        

        # Process the outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                # Filter out weak detections
                if confidence > 0.5:
                    # Get the bounding box coordinates
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])

                    # Draw the bounding box and label
                    if classes[class_id] == 'cell phone':
                        print('Cell')
                        cellphone_cheat = True
                        # cv2.putText(image, "You are found using the phone, disqualified!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 6)
                        cv2.rectangle(image, (center_x - w // 2, center_y - h // 2), (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)
                        cv2.putText(image, classes[class_id], (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if check == False:
            if face_match:
                check = True
                checkcount=checkcount+1
            else:
                cv2.putText(image, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

        if check == True and cheat == False and cellphone_cheat == False and checkcount<=1:
            cv2.putText(image, "MATCH!", (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            cv2.putText(image, "You can proceed!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        # elif cheat == True:
        #     cv2.putText(image, "You are disqualified for the exam!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
            # break
        elif cellphone_cheat == True :
            cv2.putText(image, "You are found using the phone", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            cv2.putText(image, "Disqualified!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
            flag = True
            flag_event.set()
            print("rec stoppped due to cell phone found")
            cap.release()
        if num_faces > 1:
            warnings += 1
            if warnings > 5:
                cheat = True
            cv2.putText(image, "Warning: Multiple faces detected!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        
        
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []
        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])       

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                x = angles[0] * 360
                y = angles[1] * 360

                if y < -10 or y > 10:
                    X_AXIS_CHEAT += 1
                    time_left += 1
                    time_right = 0
                    time_down = 0
                elif x < -10:
                    Y_AXIS_CHEAT += 1
                    time_down += 1
                    time_left = 0
                    time_right = 0
                elif y >= -10 and y <= 10 and x >= -10:
                    # Resetting time counters when head is neutral
                    time_left = 0
                    time_right = 0
                    time_down = 0
                    neutral_count += 1
                    if neutral_count >= neutral_threshold:
                        neutral_count = 0
                        X_AXIS_CHEAT = 0
                        Y_AXIS_CHEAT = 0

                nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                cv2.line(image, p1, p2, (255, 0, 0), 2)
                text = "Looking Left" if y < -10 else "Looking Right" if y > 10 else "Looking Down" if x < -10 else "Forward"
                text = f"{int(x)}::{int(y)} {text}"
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Head Pose Estimation', image)

        # Check if head pose exceeds threshold and trigger alarm
        if time_left > time_threshold or time_right > time_threshold or time_down > time_threshold:
            cheat = True
            print("Triggering alarm...")
            play_alarm()
        if cv2.waitKey(5) & 0xFF == 27:
            flag = True
            flag_event.set()
            print("rec stoppped")
            break
    cap.release()
    cv2.destroyAllWindows()

def play_alarm():
    # Play the alarm sound
    print("ALARM!")
    data, samplerate = sf.read(ALARM_SOUND)
    sd.play(data, samplerate, blocking=True)

def read_audio():
    p = pyaudio.PyAudio()
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 2
    fs = 44100  # Record at 44100 samples per second
    i = 0

    global flag_event  # Flag to indicate whether to stop recording
    print("rec")
    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    filename = 'record' + str(i) + '.wav'
    frames = []  # Initialize array to store frames
    
    while not flag_event.is_set():
        data = stream.read(chunk)
        frames.append(data)
        if keyboard.is_pressed('esc'): 
            print('pressed') # Check for ESC key press
            flag_event.set()  # Set the flag to stop recording
        

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert the recorded audio to text
    convert(i)

# Function to convert recorded audio to text
def convert(i):
    sound = 'record' + str(i) + '.wav'
    r = sr.Recognizer()
    
    with sr.AudioFile(sound) as source:
        r.adjust_for_ambient_noise(source)
        print("Converting Audio To Text and saving to file..... ") 
        audio = r.listen(source)
        
    try:
        value = r.recognize_google(audio)  # API call to Google for speech recognition
        os.remove(sound)
        
        if isinstance(value, bytes): 
            result = value.decode("utf-8")
        else: 
            result = value

        with open("test.txt", "w") as f: 
            f.write(result + " ") 
        
    except sr.UnknownValueError:
        print("")
    except sr.RequestError as e:
        print("{0}".format(e))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    t1 = th.Thread(target=pose)
    t2 = th.Thread(target=read_audio)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

file = open("test.txt")
data = file.read()
file.close()

# Tokenize and remove stop words from student speech
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(data.lower())  # Convert to lowercase
filtered_sentence = [w for w in word_tokens if not w in stop_words]

# Write filtered student speech to final file
with open('final.txt', 'w') as f:
    for ele in filtered_sentence:
        f.write(ele + ' ')

# Read question file
file = open("paper.txt")
data = file.read()
file.close() 

# Tokenize and remove stop words from questions
word_tokens = word_tokenize(data.lower())  # Convert to lowercase
filtered_questions = [w for w in word_tokens if not w in stop_words]

# Function to find common elements between two lists
def common_member(a, b):
    a_set = set(a)
    b_set = set(b)

    # Check length
    if len(a_set.intersection(b_set)) > 0:
        return(a_set.intersection(b_set))
    else:
        return([])

# Find common elements between questions and student speech
comm = common_member(filtered_questions, filtered_sentence)
print('Number of common elements:', len(comm))
print(comm)
