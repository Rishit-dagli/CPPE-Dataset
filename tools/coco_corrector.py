import json

num_test_images = 29
num_train_image = 1000

data = json.load(open("data/annotations/test.json"))
for i in range(num_test_images):
    data["images"][i]["file_name"] = data["images"][i]["file_name"][20:]
with open("data/annotations/test.json", "w") as jsonFile:
    json.dump(data, jsonFile)

data = json.load(open("data/annotations/train.json"))
for i in range(num_train_image):
    data["images"][i]["file_name"] = data["images"][i]["file_name"][20:]
with open("data/annotations/train.json", "w") as jsonFile:
    json.dump(data, jsonFile)
