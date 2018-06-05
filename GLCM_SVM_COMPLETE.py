import cv2

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.Extractors.GLCM import GLCM
import numpy as np

# data_set = np.loadtxt("DATABASE.txt", delimiter=" ")
# data_set[data_set == 1] = 31
# att_dataset = []
# for i in data_set:
#     oGLCM = GLCM(np.reshape(i[:-1], (35, 35)), 5)
#     oGLCM.generateCoOccurenceHorizontal()
#     oGLCM.normalizeCoOccurence()
#     oGLCM.calculateAttributes()
#     att_dataset.append(oGLCM.exportToClassfier("CLASS-" + str(i[-1])))
#     np.savetxt("GLCM_5B.txt", np.array(att_dataset), delimiter=",", fmt="%s")

atts = np.loadtxt("GLCM_8B.txt", delimiter=",", usecols=(x for x in range(24)))
labels = np.loadtxt("GLCM_8B.txt", delimiter=",", usecols=(-1), dtype=object)
labels[labels == 'CLASS-255.0'] = 'CLASS-1.0'
lista = np.unique(labels)

data_set = DataSet()
for j, i in enumerate(atts):
    data_set.addSampleOfAtt(np.array(list(i) + [labels[j]]))
data_set.atributes = data_set.atributes.astype(float)
data_set.normalizeDataSet()
quantidade_por_classe = [len(labels[labels == x]) for x in data_set.labelsNames]

for it in range(4):
    data = Data(10, 1676, samples=3352)
    data.randomTrainingTestByPercent(np.array(quantidade_por_classe).copy(), 0.5)
    data.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0, k_fold=2)

    svm = cv2.SVM()
    svm.train_auto(np.float32(data_set.atributes[data.Training_indexes]),
                   np.float32(data_set.labels[data.Training_indexes]), None, None, params=data.params)
    results = svm.predict_all(np.float32(data_set.atributes[data.Testing_indexes]))

    data.setResultsFromClassfier(results, data_set.labels[data.Testing_indexes])
    data_set.append(data)
print data_set
print "fim"
