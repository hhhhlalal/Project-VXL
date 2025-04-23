import os
import cv2
import numpy as np
from PIL import Image

# Đường dẫn nguồn và nơi lưu
path = 'D:/Project VXL/dataset'
path_save = 'D:/Project VXL/dataset_save'

# Kích thước resize
newsize = (200, 200)

# Duyệt toàn bộ thư mục con
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, path)  # Lấy đường dẫn tương đối
                save_path = os.path.join(path_save, rel_path)

                # Tạo thư mục con trong thư mục đích nếu chưa có
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Đọc và resize ảnh
                pil_img = Image.open(input_path).convert('RGB')
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                resized_img = cv2.resize(img, newsize, interpolation=cv2.INTER_AREA)

                # Ghi ảnh
                success = cv2.imwrite(save_path, resized_img)
                if success:
                    print(f"✅ Đã resize và lưu: {save_path}")
                else:
                    print(f"❌ Không thể lưu: {save_path}")

            except Exception as e:
                print(f"⚠️ Lỗi xử lý ảnh {file}: {e}")

print("\n🎉 Hoàn tất resize ảnh!")
