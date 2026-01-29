import os
import json


def get_class_information(json_file_path, json_type='db_class_info'):
    label_map = dict()
    classes = set()

    # read json data
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    if json_type == 'db_class_info':
        # get label_map and classes
        for one_dict in json_data['Class']:
            for key, value in one_dict.items():
                label_map[int(key)] = int(value)
                classes.add(str(value).zfill(7))
    elif json_type == 'annotation':
        for category in json_data['categories']:
            name = category['name']
            category_id = category['id']
            label_value = name.split('_')[-1]

            classes.add(name)
            label_map[int(category_id)] = int(category_id)

    return label_map, tuple(classes)


def re_build_all_config(cfg, new_label_map, new_classes):
    # model
    cfg.num_classes = len(new_classes) + 1

    # dataset
    cfg.dataset.class_names = new_classes
    cfg.dataset.label_map = new_label_map

# MSP : re label_map
# import json, os
# json_path = "./data/coco/annotations/instances_train2017.json"
##json_path = "coco/instances_train2017.json"
# print(f"====> json_path = {json_path},  {os.path.exists(json_path)}")
# with open(json_path, 'r') as json_file:
#    json_data = json.load(json_file)
#
# COCO_CLASSES = set()
# COCO_LABEL_MAP = dict()
#
# for category in json_data['categories']:
#    name = category['name']
#    category_id = category['id']
#    label_value = name.split('_')[-1]
#
#    COCO_CLASSES.add(name)
#    #COCO_LABEL_MAP[int(category_id)] = int(label_value)
#    COCO_LABEL_MAP[int(category_id)] = int(category_id)
#
# COCO_CLASSES = tuple(COCO_CLASSES)
# print(f"=====> COCO_CLASSES = {COCO_CLASSES}")
# print(f"=====> COCO_LABEL_MAP = {COCO_LABEL_MAP}")


