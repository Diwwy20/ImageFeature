from fastapi import FastAPI, Request
import numpy as np
import cv2
import base64

app = FastAPI()

@app.get("/")
def root(): 
    return {"message": "This is my api"}

@app.get("/api/genhog")
async def HOG(data: Request):
    try:
        json = await data.json()
        img_data = json["img_base64"] # key ของ json
        # ตัด url ให้อยู่ในรูป base64 
        data_split = img_data.split(',', 1)
        img_str = data_split[1]

        # ทำการถอดรหัส ภาพที่ถูกเข้ารหัสไว้ จะเอาภาพต้นฉบับที่ใช้งานได้
        decode_image_data = base64.b64decode(img_str) # แปลงให้เป็น binary data

        # แปลงข้อมูลที่ถอดรหัสเป็นรูปภาพ
        decode_img = cv2.imdecode(np.frombuffer(decode_image_data, np.uint8), cv2.IMREAD_GRAYSCALE) 

        # ปรับขนาดไซส์ภาพให้เล็กลงนิสนุงง
        resized_img = cv2.resize(decode_img, (128, 128), cv2.INTER_AREA)

        win_size = resized_img.shape
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9

        # Set the parameters of the HOG descriptor using the variables
        # defined above
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
        cell_size, num_bins)

        # คำสั่งนี้จะได้ numpy array มา
        hog_descriptor = hog.compute(resized_img)
       
        # เปลี่ยนข้อมูลที่ได้จาก numpy array ให้เป็น list เพื่อง่ายต่อการสร้าง json
        # ค่าที่ได้หลังจากทำเป็น list จะเป็นตัวเลขหรือค่าที่บ่งชี้ลักษณะของภาพนั้น 
        hog_descriptor_list = hog_descriptor.flatten().tolist()
        return {"HOG": hog_descriptor_list}
    
    except Exception as ex:
         return {"error": f"เกิดข้อผิดพลาด: {str(ex)}"}