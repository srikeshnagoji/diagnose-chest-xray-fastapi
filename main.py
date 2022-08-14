from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from io import BytesIO
import uvicorn

app = FastAPI(title='Hello world')
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model():
    model = tf.keras.models.load_model('DenseNet121_model.h5')
    print("Model loaded.")
    return model

model = load_model()

classes = ("COVID", "NORMAL","PNEUMONIA")
CATEGORIES = sorted(classes)

def decode_predictions(predictions):
    return CATEGORIES[np.argmax(predictions)];

def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_CUBIC)
    return frame

def predict(image: Image.Image):
    image = np.expand_dims(image, 0)
    result = decode_predictions(model.predict(image/255))
    # response = []
    # for i, res in enumerate(result):
    #     resp = {}
    #     resp["class"] = res[1]
    #     resp["confidence"] = f"{res[2]*100:0.2f} %"
    #     response.append(resp)
    # return response
    return result

def read_imagefile(file) -> Image.Image:
    return Image.open(BytesIO(file))

@app.get('/')
async def hello_world():
    return "hi there! Wassup!"


@app.post("/predict/images")
async def predict_api(files: list[UploadFile]):
    for file in files:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if not extension:
            return "Image must be jpg or png format!"
    
    predictions = dict()

    for file in files:
        image = load_image_into_numpy_array(await file.read())
        prediction = predict(image)
        predictions[file.filename] = prediction
    
    return predictions


if __name__ == "__main__":
    uvicorn.run(app)