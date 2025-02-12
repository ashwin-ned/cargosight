import os
import json





def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
        intrinsics = data["intrinsics"]
        print(intrinsics)
        fx = intrinsics[0]
        fy = intrinsics[4]
        cx = intrinsics[2]
        cy = intrinsics[5] 

    return fx, fy, cx, cy

def read_data(folder_path):
    global Fx
    global Fy
    global Cx
    global Cy
    global S
    Fx = 0.0
    Fy = 0.0
    Cx = 0.0
    Cy = 0.0
    S = 0
    for file in os.listdir(folder_path):
        if file.endswith(".json") and file.startswith("frame"):
            S += 1
            json_path = os.path.join(folder_path, file)
            fx, fy, cx, cy = read_json(json_path)
            Fx += fx
            Fy += fy
            Cx += cx
            Cy += cy
       

if __name__ == "__main__":

    # Example usage Navvis Data
    folder_path = "/home/admin-anedunga/Desktop/benchmarking_data/ten-container-dataset/"
    output_path = "/home/admin-anedunga/Desktop/benchmarking_data/"
    for root, dirs, files in os.walk(folder_path):
        for directory in dirs:
            print(f"Reading files in directory:{directory}")
            read_data((os.path.join(folder_path, root, directory)))
    print(f"Average Fx: {Fx/S}, Fy: {Fy/S}, Cx: {Cx/S}, Cy: {Cy/S}")
