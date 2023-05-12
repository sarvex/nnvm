import os
import sys

FOLDERS = ["core", "pass", "c_api"]

with open(sys.argv[1], "w") as fo:
    for folder in FOLDERS:
        path = str(os.path.join("../src", folder))
        flst = os.listdir(path)
        for f in flst:
        	if f.endswith(".cc") == True:
            	fo.write('#include "' + str(os.path.join("src", folder, f)) + '"\n')
