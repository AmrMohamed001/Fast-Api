from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow import keras  # Updated import
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Load the saved model
model = keras.models.load_model("https://media.githubusercontent.com/media/AmrMohamed001/Fast-Api/master/knee_xray_Xceptionnet_complete_model.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    img = img.resize((224, 224))
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array = img_array / 255.0  # Normalize
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return JSONResponse(content={"predicted_label": int(predicted_class[0])})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
