import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# load mobilenetv2 model
model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# open webcamdata
capture = cv2.VideoCapture(0)

while True:
    # capture frame from webcam
    ret, frame = capture.read()

    # resize frame for mobilenetv2 model
    frame = cv2.resize(frame, (224, 224))

    # preprocess frame 
    preprocess_frame = preprocess_input(np.expand_dims(frame, axis=0))

    # do predictions/get inference
    predictions = model.predict(preprocess_frame)

    # get highest confidence prediction
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    class_id, class_label, confidence = decoded_predictions[0]

    # print label and confidence level on frame
    text = f"{class_label}: {confidence:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # display frame
    cv2.imshow('Object Recognition', frame)

    # export frame and save as file
    #cv2.imwrite('output_image.jpg', frame)

    # press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# stop video capture
capture.release()
cv2.destroyAllWindows()
