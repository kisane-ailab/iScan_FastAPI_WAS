import os
import json
import hashlib

def validate_checksum(base_dir, checksum_filename):
    version = "0.0.0"
    model_integrity = "none"
    fail_list = []
    with open(base_dir + checksum_filename) as json_file:
        json_data = json.load(json_file)
        if "version" in json_data:
            version = json_data["version"]
        '''if "hash_cnt" in json_data and json_data["hash_cnt"]:
            if "hashlist" in json_data:
                is_pass = True
                for filename, old_hash in json_data["hashlist"].items():
                    sha256 = hashlib.sha256()
                    filepath = base_dir + filename
                    with open(filepath, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            sha256.update(chunk)
                    if sha256.hexdigest() != old_hash:
                        is_pass = False
                        fail_list.append(filepath)
                model_integrity = "pass" if is_pass else "fail"'''
        #else:
        #    model_integrity = "fail"
    return version, model_integrity, fail_list

def calculate_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def calculate_checksum(version, base_dir):
    hash_info = {}
    hash_info["version"] = version
    hash_info["hash_algorithm"] = "SHA256"
    hash_info["hashlist"] = {}

    filelist = ["best_model.pth", "best_config.py", "Yolof_Class_Info.json", "db_class_info.json",
                "yolof_rgb_fp16.onnx", "yolof_rgb_fp16.trt", "yolof_rgb_onnx_config.py", "yolof_rgb_trt_config.py"]
    
    hash_cnt = 0
    for filename in filelist:
        fpath = base_dir + filename
        if os.path.exists(fpath):
            hash_info["hashlist"][filename] = calculate_sha256(fpath)
            hash_cnt += 1
    hash_info["hash_cnt"] = hash_cnt

    return hash_info


if __name__=="__main__":
    path_to_check = "/home/nvidia/JetsonMan/resource/Artis_AI/checkpoints/"
    '''path_to_check = "C:/Users/Kisan/Downloads/푸디푸디/update_Artis_AI_Model/"
    model_version = "0.0.20"'''
    '''path_to_check = "C:/Users/Kisan/Downloads/일리에/update_Artis_AI_Model/"
    model_version = "0.1.5"'''
    path_to_check = "C:/Users/Kisan/Downloads/천상가옥/update_Artis_AI_Model/"
    model_version = "0.1.6"
    hash_info = calculate_checksum(model_version, path_to_check)
    with open(path_to_check + "version.json", "w") as file:
        json.dump(hash_info, file, indent=4)