import cv2
import numpy as numpy

#XML = eXtensible Markup Language: : lưu trữ thông dữ liệu chuyển đổi
faceDetect = cv2.CascadeClassifier('Face-Recognition-Using-SVM-master/haarcascade_frontalface_default.xml')

#Tạo đối tượng quay video, đối tượng này sẽ giúp phát trực tuyến hoặc hiển thị video., Mở Camera
#1 được sử dụng để chọn các máy ảnh khác nhau nếu bạn có nhiều máy ảnh được đính kèm. 
#Theo mặc định 0 là số chính của bạn.
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
print(rec.read('Face-Recognition-Using-SVM-master/trainer.yml'))
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    #Mở Camera
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #1.3 là scaleFactor – càng nhỏ thì càng phát hiện tốt hơn, nhưng thuật toán chạy lâu hơn
    #5 là minNeighbors – càng lớn thì phát hiện được càng ít, nhưng khuôn mặt tìm được có độ chính xác và chất lượng cao hơn.
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    #Dectect : tim duoc nhieu kich thuoc khuon mat
    #MultiScale: tìm được các khuôn mặt từ xa đến bé
    for (x, y, w, h) in faces:
            # hình ảnh: Đây là hình ảnh mà hình chữ nhật sẽ được vẽ.
            # start_point: Là tọa độ bắt đầu của hình chữ nhật.
            # end_point: Là tọa độ cuối của hình chữ nhật.
            # color: Là màu của đường viền của hình chữ nhật cần vẽ
            # Độ dày: Là độ dày của đường viền hình chữ nhật tính bằng px.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        #Hàm Predict dự đoán và thống kê 
        #Confidence: dựa trên số từ confidence nào lớn hơn sẽ gán cho label đó
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        #print(conf)
        if id == 1:
            cv2.putText(img, "thai", (x ,y + h), font, 4,(255,255,255), 2, cv2.LINE_AA)   
            # img : ảnh hiển thị
            # text: nội dung chữ.
            # org: tọa độ đặt chữ lên ảnh, nó sẽ ứng với góc dưới bên trái của "text" (bottom-left).
            # fontFace: font chữ
            # fontScale: cỡ chữ
            # color: màu sắc chữ theo hệ màu BGR.
            # thickness: độ dày (đậm) của chữ. Trong word ta hay gọi là bold.
            # lineType: độ "smooth" của nét.
            # bottomLeftOrigin: dùng hệ trục ở góc trái hay không? Set False tức dùng gốc tọa độ ở góc phía trên bên trái ảnh, đây là gốc tọa độ chuẩn trong Xử Lý Ảnh.
        pass
    cv2.imshow("Hien Thi Khuon Mat", img)
    if(cv2.waitKey(1) == ord('q')):
        break
    pass
# Giai phong cammera khi thuc hien xong thao tac
cam.release()
# Đóng tất cả cửa sổ
cv2.destroyAllWindows()