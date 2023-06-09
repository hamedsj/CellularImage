# Cellular Image

Recreate an image with many other images as cells
|         Input         |        output         |
|-----------------------|-----------------------|
| <img src="https://github.com/hamedsj/CellularImage/assets/17751865/0eaa0925-686a-41c2-97a1-466d7f596ffb" width="300" height="300" />  | <img src="demo-files/output-4-hr-200-demo.png" width="300" height="300" /> |


### Built With
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)


### Prerequisites

All the modules you need for running the project are listed in the `requirements.txt` you can install them easily using this command:

```bash
pip install -r requirements.txt
  ```

## Usage
1. Put your [Unsplash API ACCESS_TOKEN](https://unsplash.com/oauth/applications) in te sample.env file
2. You can simply run the application using command below:
```bash
python3 cellular-images.py -i YOUR_IMAGE_PATH
```

There is also some other flags that you can use in your command:

* <em>**-h**</em>

You can use this flag for seeing the help menu. use it like:
```bash
python3 cellular-images.py -h
```


* <em>**--output**</em> or <em>**-o**</em>

Output path with png format. Its default value is `./output.png`. sample:
```bash
python3 cellular-images.py -i YOUR_IMAGE_PATH -o OUTPUT_IMAGE_PATH
```

* <em>**--output_size**</em> or <em>**-os**</em>

Number of cell images used in every row of output image. Its default value is 200. sample:
```bash
python3 cellular-images.py -i YOUR_IMAGE_PATH -o OUTPUT_IMAGE_PATH -os 100
```

* <em>**--cell_size**</em> or <em>**-cs**</em>

Size of cell images in the output image. Its default value is 50. sample:
```bash
python3 cellular-images.py -i YOUR_IMAGE_PATH -o OUTPUT_IMAGE_PATH -os 100 -cs 75
```

## License

Copyright [2023] [HamidReza Shajaravi]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

