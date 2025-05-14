import os
import cv2
import numpy as np

def apply_sepia_filter(image):
    sepia_kernel = np.array([
        [0.131, 0.534, 0.272],  # B
        [0.168, 0.686, 0.349],  # G
        [0.189, 0.769, 0.393]   # R
    ])
    sepia_image = cv2.transform(image, sepia_kernel)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

def create_sepia_dataset(input_dir="input_images", output_dir="sepia_dataset"):
    normal_dir = os.path.join(output_dir, "normal")
    sepia_dir = os.path.join(output_dir, "sepia")
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(sepia_dir, exist_ok=True)

    normal_count = 0
    sepia_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            image = cv2.imread(img_path)

            # Salvează imaginea originală
            normal_path = os.path.join(normal_dir, f"normal_{normal_count}.jpg")
            cv2.imwrite(normal_path, image)
            normal_count += 1

            # Aplică filtrul sepia direct pe imaginea BGR
            sepia_image = apply_sepia_filter(image)
            sepia_path = os.path.join(sepia_dir, f"sepia_{sepia_count}.jpg")
            cv2.imwrite(sepia_path, sepia_image)
            sepia_count += 1

            if normal_count % 10 == 0:
                print(f"Procesate {normal_count} imagini...")

    print(f"Imagini normale: {normal_count}")
    print(f"Imagini sepia: {sepia_count}")
    print(f"Total: {normal_count + sepia_count}")
    return normal_count, sepia_count

def main():
    if not os.path.exists("input_images"):
        return
    create_sepia_dataset()

if __name__ == "__main__":
    main()
