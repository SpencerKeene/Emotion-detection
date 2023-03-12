from multiprocessing import Process, Array
import socket
import argparse

HOST = '127.0.0.1'
PORT = 34012

def emotion_detection_process(emotion_state, mode):
    """
    Child process for real-time detection of player's emotion or training of the emotion detection model
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import os
    import time

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # plots accuracy and loss curves
    def plot_model_history(model_history):
        """
        Plot Accuracy and Loss curves given the model_history
        """
        fig, axs = plt.subplots(1,2,figsize=(15,5))
        # summarize history for accuracy
        axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
        axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
        axs[0].legend(['train', 'val'], loc='best')
        # summarize history for loss
        axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
        axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
        axs[1].legend(['train', 'val'], loc='best')
        fig.savefig('plot.png')
        plt.show()

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    # If you want to train the same model or try other models, go for this
    if mode == "train":
        # Define data generators
        train_dir = 'data/train'
        val_dir = 'data/test'

        num_train = 28709
        num_val = 7178
        batch_size = 64
        num_epoch = 50

        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=(48,48),
                batch_size=batch_size,
                color_mode="grayscale",
                class_mode='categorical')

        validation_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=(48,48),
                batch_size=batch_size,
                color_mode="grayscale",
                class_mode='categorical')

        # Train
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
        model_info = model.fit_generator(
                train_generator,
                steps_per_epoch=num_train // batch_size,
                epochs=num_epoch,
                validation_data=validation_generator,
                validation_steps=num_val // batch_size)
        plot_model_history(model_info)
        model.save_weights('model.h5')

    # emotions will be displayed on your face from the webcam feed
    elif mode == "display":
        model.load_weights('model.h5')

        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)

        # dictionary which assigns each label an emotion (alphabetical order)
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # create store of past emotions
        class EmotionStore:
            class Emotion:
                def __init__(self, emotion_string):
                    self.emotion_string = emotion_string
                    self.time_recorded = time.time()
                
                def is_outdated(self, max_age_seconds):
                    now = time.time()
                    return now > self.time_recorded + max_age_seconds
                
            def __init__(self):
                self.past_emotions = []

            def add(self, emotion_string):
                self.past_emotions.append(self.Emotion(emotion_string))
            
            def remove_outdated(self, max_age_seconds = 1.0):
                """ Deletes any stored emotions that are considered outdated by the given maximum age """
                not_outdated = lambda emotion: not(emotion.is_outdated(max_age_seconds))
                self.past_emotions = list(filter(not_outdated, self.past_emotions))
            
            def remove_oldest(self, max_len = 50):
                """ Deletes the oldest stored emotions to limit the store's size to a given maximum length """
                if len(self.past_emotions) > max_len:
                    self.past_emotions = self.past_emotions[-max_len:]

            def get_mode(self):
                past_emotion_strings = [e.emotion_string for e in self.past_emotions]
                mode = max(set(past_emotion_strings), key=past_emotion_strings.count)
                return mode

        emotion_store = EmotionStore()

        # start the webcam feed
        cap = cv2.VideoCapture(0)
        while True:
            # Find haar cascade to draw bounding box around face
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                emotion_string = emotion_dict[maxindex]
                cv2.putText(frame, emotion_string, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Update emotion store
                emotion_store.add(emotion_string)
                # emotion_store.remove_outdated()
                emotion_store.remove_oldest()

                # Update the current emotion used by the server process
                with emotion_state.get_lock():
                    mode_emotion = emotion_store.get_mode()
                    print(f'===== setting emotion: {mode_emotion} -> {str.encode(mode_emotion)}')
                    emotion_state.value = str.encode(mode_emotion)

            cv2.imshow('Video', cv2.resize(frame,(800,480),interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def emotion_server_process(emotion_state):
    """
    Child process for creating a server to broadcast the current player's emotion
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f'listing on port {PORT}')
        while True:
            conn, addr = s.accept()
            # When client asks for emotion, send emotion as utf-8 string
            with conn, emotion_state.get_lock():
                print(f"Connected by {addr}, sending {emotion_state.value}")
                conn.sendall(emotion_state.value)
                conn.listen
                


if __name__ == '__main__':
    # initialize emotion as a string at least as long as the longest emotion
    emotion_state = Array('c', str.encode('xxxxxxxxx'))

    # command line argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",help="train/display")
    mode = ap.parse_args().mode

    # Start the emotion detection process for handling real-time detection of player's emotion or training of model
    p_emotion_detection = Process(target=emotion_detection_process, args=[emotion_state, mode])
    p_emotion_detection.daemon = True
    p_emotion_detection.start()

    # If mode is 'display', start the server process for handling sending recognized emotion to client processes
    if mode == "display":
        p_emotion_server = Process(target=emotion_server_process, args=[emotion_state])
        p_emotion_server.daemon = True
        p_emotion_server.start()

    # Loop infinitely to keep program running
    # This makes it easier to kill child processes in the terminal by just killing this process with CTRL+C
    while True:
        pass