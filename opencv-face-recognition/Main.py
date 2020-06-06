from live import indexer
from Recognizer import indexe
# import os
# import glob
# #first we will run camera and take picture.

grab = indexer()
print(grab)

final = indexe(grab)
print(final)


# #Then the recognizer will run and return true/false.
#exec(open("./Recognizer.py").read())
# print("successful")

import os

def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)

#print(newest('C:\\Users\\UN\\Desktop\\upload_file_python-master\\src\\images\\'))
