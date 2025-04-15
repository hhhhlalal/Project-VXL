import os
import cv2
import numpy as np
from PIL import Image

#Nhận dạng mặt người
recognizer = cv2.face.LBPHFaceRecognizer_create()
# đường dẫn để set ảnh
path = 'D:/Project VXL/dataset'
#đường dẫn để lưu ảnh đã set
path_save = 'D:/Project VXL/dataset_save'


#trả về một danh sách chứa tên của các mục trong thư mục được cung cấp bởi đường dẫn
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
faces = []
IDs = []
for imagePath in imagePaths:
    faceImg = Image.open(imagePath).convert('L')
    
    faceNp= np.array(faceImg, 'uint8') #uint8 la Số nguyên không dấu (0 đến 255)
    
    print('===============================imagePath=======================')
    print(imagePath)
    vitri = imagePath.find('\\')
    
    #dem vi tri cua anh va in ra vi tri
    only_filename=imagePath[vitri + 1:len(imagePath)]
    print(only_filename)
    
    imagePath = imagePath.replace("\\","/")
    print('===============================imagePath=======================')
    print(imagePath)
    img = cv2.imread(imagePath)

    print(img.shape)# lấy ra kích thước của mảng này với h, w, d lần lượt là chiều cao, chiều rộng, độ sâu của bước ảnh

    newsize = (200, 200)
    # INTER_AREA: sử dụng quan hệ vùng pixel để lấy mẫu lại. Phương pháp phù hợp để giảm kích thước của hình ảnh (thu nhỏ).
    im1 = cv2.resize(img, newsize, interpolation = cv2.INTER_AREA)  
    filename = path_save +'/' + only_filename
    print('===============================filename=======================')
    #Phương thức image.shape trong Python trả về ba giá trị: Chiều cao, chiều rộng và số kênh.
    print(im1.shape)
    print(filename)

    cv2.imwrite(filename, im1)

print('Xong roi!')


