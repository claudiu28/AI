import pickle
import os
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def extract_images_batch(batch_name, output_dir, numbers=100):

    os.makedirs(output_dir, exist_ok=True)

    batch_path = os.path.join('cifar-10/cifar-10-batches-py', batch_name)
    batch_dict = unpickle(batch_path)

    images = batch_dict[b'data']
    labels = batch_dict[b'labels']

    # 10000x3072 -> 10000x32x32x3
    images = images.reshape(len(images), 3, 32, 32).transpose(0, 2, 3, 1)

    for i in range(min(numbers, len(images))):
        image_rgb = images[i]
        image_upscaled = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        filename = f"{batch_name}_img_{i}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(image_upscaled, cv2.COLOR_RGB2BGR))
        if (i + 1) % 10 == 0:
            print(f"Extrase {i + 1} imagini din {batch_name}")
    return min(numbers, len(images))


def extract_all_batches(output_dir="input_images", images_per_batch=100):
    batch_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch"
    ]

    total_images = 0

    for batch_file in batch_files:
        count = extract_images_batch(batch_file, output_dir, images_per_batch)
        total_images += count
    return total_images


def main():
    if not os.path.exists('cifar-10/cifar-10-batches-py'):
        return

    extract_all_batches(output_dir="input_images", images_per_batch=100)

    if os.path.exists("input_images"):
        image_count = len([f for f in os.listdir("input_images") if f.endswith('.jpg')])
        print(f"Total imagini folder: {image_count}")

if __name__ == "__main__":
    main()