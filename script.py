
import os #مكتبة للتعامل مع الملفات والمجلدات
import cv2 # مكتبة قوية للتعامل مع صور
import json # json مكتبة للتعامل مع ملفات 

images_path = "images" # نعرف مكان الصور الاصلية
ann_path = "annotations" # json نعرف مكان ملفات 

out_mask = "dataset_cnn/with_mask" # ونحدد مسارة  with_mask   ملف باسم 
out_nomask = "dataset_cnn/without_mask" # ونحدد مسارة  without_mask   ملف باسم

os.makedirs(out_mask, exist_ok=True) # with_mask هنا يتم انشاء ملف 
os.makedirs(out_nomask, exist_ok=True) # without_mask هنا يتم انشاء ملف 

#  يعني لو المجلد موجود لا يعطي خطأ exist_ok=True معلومة 


MASK_CLASSES = ["face_with_mask", "mask_surgical"]# json الكلاسات التي تعتبر بكمامة من ملف 
NO_MASK_CLASSES = ["face_no_mask"]# json الكلاسات التي تعتبر بدون كمامة من ملف 

for file in os.listdir(ann_path):
    if file.endswith(".json"):
        with open(os.path.join(ann_path, file), encoding="utf-8") as f:
            data = json.load(f)

        img_name = data["FileName"]
        img_path = os.path.join(images_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ لم يتم إيجاد الصورة: {img_name}")
            continue

        h, w = img.shape[:2]

        for i, obj in enumerate(data["Annotations"]):
            cls = obj["classname"]
            x1, y1, x2, y2 = obj["BoundingBox"]

            # تأكد أن القيم داخل الصورة
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # تصنيف الحفظ
            if cls in MASK_CLASSES:
                save_path = os.path.join(out_mask, f"{img_name}_mask_{i}.jpg")
            elif cls in NO_MASK_CLASSES:
                save_path = os.path.join(out_nomask, f"{img_name}_nomask_{i}.jpg")
            else:
                continue  # تجاهل أي كلاس آخر

            cv2.imwrite(save_path, face)

print("🎉 تم تجهيز بيانات CNN بنجاح!")
