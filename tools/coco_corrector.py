import json

num_test_images = 29
num_train_image = 1000

data = json.load(open("coco/test.json"))
for i in range(num_test_images):
    data["images"][i]["file_name"] = data["images"][i]["file_name"][20:]
with open("coco/test.json", "w") as jsonFile:
    json.dump(data, jsonFile)

data = json.load(open("coco/train.json"))
for i in range(num_train_image):
    data["images"][i]["file_name"] = data["images"][i]["file_name"][20:]
with open("coco/train.json", "w") as jsonFile:
    json.dump(data, jsonFile)
