from tqdm import tqdm
import requests
import os
import json
import numpy as np
import cv2
from dotenv import load_dotenv
import argparse

# import cv2.cv2 as cv2

load_dotenv()

unsplash_access_token = os.getenv("UNSPLASH_ACCESS_KEY")


class UnsplashImage:

    def __init__(self, image_id: str, url: str, local_path: str = None, dominant_gray_color: int = None):
        self.image_id = image_id
        self.url = url
        self.local_path = local_path
        self.dominant_gray_color = dominant_gray_color

    def __str__(self):
        return self.url

    def __repr__(self):
        return str(self.dominant_gray_color)


def fetch_page_of_images(page: int = 1, per_page: int = 100):
    response_cache_file_name = f"./unsplash-responses/response-cache-{page}-{per_page}.json"
    response_string = ""
    try:
        if not os.path.exists(response_cache_file_name):
            print("calling unsplash api --> start")
            response = requests.get(url="https://api.unsplash.com/photos",
                                    params=[("page", str(page)),
                                            ("per_page", str(per_page)),
                                            ("client_id", unsplash_access_token)])
            print(f"calling unsplash api --> end --> {response.status_code}")
            if response.status_code != 200:
                return None
            file = open(response_cache_file_name, "w")
            file.write(response.text)
            response_string = response.text
        else:
            file = open(response_cache_file_name, "r")
            response_string = file.read()

        response_json = json.loads(response_string)
        result = [UnsplashImage(image_id=item["id"], url=item["urls"]["small"]) for item in response_json]
        return result
    except Exception as e:
        print(str(e))
        return None


def fetch_images():
    images = []
    for page_index in range(10):
        page_images = fetch_page_of_images(page=page_index + 1)
        if page_images is not None:
            images += page_images
    return images


def load_image(image: UnsplashImage):
    image_path = f"./unsplash_images/{image.image_id}.jpg"
    try:
        if os.path.exists(image_path):
            image.local_path = image_path
            return True
        # print(f"Downloading new image --> {image.url}")
        response = requests.get(image.url, stream=True)
        with open(image_path, "wb") as output_image:
            output_image.write(response.content)
        image.local_path = image_path
        return True
    except:
        return False


def center_crop(image):
    shape = image.shape
    height, width = int(shape[0]), int(shape[1])
    if height == width:
        return image
    elif height > width:
        diff = int((height - width) / 2)
        return image[diff:width + diff, :]
    else:
        diff = int((width - height) / 2)
        return image[:, diff:height + diff]


def find_dominant_color_resize(image_path: str):
    original_image = center_crop(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    original_image = cv2.resize(original_image, (1, 1), interpolation=cv2.INTER_AREA)
    return original_image.flatten()[0]


def find_dominant_color_kmeans(image_path: str):
    original_image = center_crop(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
    flatten_image = np.float32(original_image.reshape(-1, 1))
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 1.0)
    _, labels, center = cv2.kmeans(flatten_image, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    dominant_color = center[np.bincount(labels.flatten()).argmax()][0]
    return dominant_color


def cluster_images_based_on_color(sorted_images: list):
    result = {}
    for image in sorted_images:
        if image.dominant_gray_color in result.keys():
            result[image.dominant_gray_color] = result[image.dominant_gray_color] + [image]
        else:
            result[image.dominant_gray_color] = [image]
    return result


def find_nearest_image(sorted_images: list, target: int):
    searching_set = list({img.dominant_gray_color for img in sorted_images})
    step = 0
    while step < max(target, 255 - target):
        if min(target + step, 255) in searching_set:
            return target + step
        if max(target - step, 0) in searching_set:
            return target - step
        step += 1
    return 0

    ###
    left = 0
    right = len(searching_set) - 1
    while left < right:
        if right - left + 1 <= 2:
            if abs(target - searching_set[right]) >= abs(target - searching_set[left]):
                return searching_set[left]
            else:
                return searching_set[right]
        middle_index = (left + (right - 1)) // 2
        if searching_set[middle_index] == target:
            return searching_set[middle_index]
        elif searching_set[middle_index] < target:
            left = middle_index + 1
        else:
            right = middle_index - 1
    return searching_set[left] if abs(searching_set[left] - target) <= abs(searching_set[right] - target) \
        else searching_set[right]


def main(args):
    if args.input is None or not os.path.exists(str(args.input)):
        print("You should pass the input path argument. use -h for help")
        return

    input_image_path = str(args.input)
    output_image_path = str(args.output)
    output_size = int(args.output_size)
    cell_image_size = int(args.cell_size)

    images = fetch_images()
    for image in tqdm(images, total=len(images)):
        loaded_successfully = load_image(image)
        if loaded_successfully:
            image.dominant_gray_color = find_dominant_color_resize(image.local_path)
        else:
            print(f"failure --> {image.id}")

    sorted_images = sorted(images, key=lambda item: item.dominant_gray_color)
    color_to_images = cluster_images_based_on_color(sorted_images)
    usage_dict = {}
    for image in sorted_images:
        usage_dict[image.dominant_gray_color] = []
    cells = []
    input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if input_image.shape[0] >= input_image.shape[1]:
        input_image = cv2.resize(input_image,
                                 (output_size, int(float(input_image.shape[1] * output_size) / input_image.shape[0])),
                                 interpolation=cv2.INTER_AREA)
    else:
        input_image = cv2.resize(input_image,
                                 (int(float(input_image.shape[0] * output_size) / input_image.shape[1]), output_size),
                                 interpolation=cv2.INTER_AREA)

    for gray_pixel in input_image.flatten():
        if gray_pixel in color_to_images.keys():
            nearest_available_color = gray_pixel
        else:
            nearest_available_color = find_nearest_image(sorted_images, gray_pixel)
        images_of_nearest_color = color_to_images[nearest_available_color]
        usage_of_nearest_color = usage_dict[nearest_available_color]
        if len(usage_of_nearest_color) == len(images_of_nearest_color):
            cells.append(usage_of_nearest_color[-1])
            if len(usage_of_nearest_color) != 1:
                new_usage = [usage_of_nearest_color[-1]] + usage_of_nearest_color[1:]
                usage_dict[nearest_available_color] = new_usage
        else:
            for image_of_nearest_color in images_of_nearest_color:
                if image_of_nearest_color not in usage_of_nearest_color:
                    cells.append(image_of_nearest_color)
                    new_usage = [image_of_nearest_color] + usage_of_nearest_color
                    usage_dict[nearest_available_color] = new_usage
                    break
    result_image_rows = []
    for row in tqdm(range(output_size), total=output_size):
        row_list = []
        for column in range(output_size):
            cell_image = center_crop(cv2.imread(cells[row * output_size + column].local_path, cv2.IMREAD_GRAYSCALE))
            cell_image = cv2.resize(cell_image, (cell_image_size, cell_image_size), interpolation=cv2.INTER_AREA)
            row_list.append(cell_image)
        result_image_rows.append(cv2.hconcat(row_list))
    result_image = cv2.vconcat(result_image_rows)
    cv2.imwrite(output_image_path, result_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input image path with jpg, jpeg or png format")
    parser.add_argument("-o", "--output", default="./output.png", help="output path with png format, default=./output.png")
    parser.add_argument("-cs", "--cell_size", default=50, help="size of cell images in the output image, default=50")
    parser.add_argument("-os", "--output_size", default=200, help="number of cell images used in every row of output image, default=200")

    main(parser.parse_args())
