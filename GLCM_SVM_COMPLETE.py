from MachineLearn.Classes.data import Data
from MachineLearn.Classes.Extractors.GLCM import GLCM
import numpy as np

data_set = np.loadtxt("DATABASE.txt", delimiter=" ")
data_set[data_set == 1] = 31
att_dataset = []
for i in data_set:
    oGLCM = GLCM(np.reshape(i[:-1], (35, 35)), 5)
    oGLCM.generateCoOccurenceHorizontal()
    oGLCM.normalizeCoOccurence()
    oGLCM.calculateAttributes()
    att_dataset.append(oGLCM.exportToClassfier("CLASS-" + str(i[-1])))
    np.savetxt("GLCM_5B.txt", np.array(att_dataset), delimiter=",", fmt="%s")
