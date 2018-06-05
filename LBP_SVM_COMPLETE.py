from MachineLearn.Classes.data import Data
from MachineLearn.Classes.Extractors.LBP import Lbp8Bits
import numpy as np

data_set = np.loadtxt("DATABASE.txt", delimiter=" ")
data_set[data_set == 1] = 255
att_dataset = []
for i in data_set:
    olbp = Lbp8Bits(np.reshape(i[:-1], (35, 35)))
    olbp.calculate_attributes()
    att_dataset.append(olbp.export_to_classifier("CLASS-" + str(i[-1])))
    np.savetxt("LBP_8B.txt", np.array(att_dataset), delimiter=",", fmt="%s")
