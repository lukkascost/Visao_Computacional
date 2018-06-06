import cv2

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment
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

experiment = Experiment()
it_for_data_set = 50
for file_index in range(5, 9):
    print "DATASET: ", file_index
    atts = np.loadtxt("GLCM_{}B.txt".format(file_index), delimiter=",", usecols=(x for x in range(24)))
    labels = np.loadtxt("GLCM_{}B.txt".format(file_index), delimiter=",", usecols=(-1), dtype=object)
    labels[labels == 'CLASS-{}.0'.format((2 ** file_index) - 1)] = 'CLASS-1.0'

    data_set = DataSet()
    for j, i in enumerate(atts):
        data_set.addSampleOfAtt(np.array(list(i) + [labels[j]]))
    data_set.atributes = data_set.atributes.astype(float)
    data_set.normalizeDataSet()
    quantidade_por_classe = [len(labels[labels == x]) for x in data_set.labelsNames]
    for it in range(it_for_data_set):
        print "\t", it
        data = Data(10, 1676, samples=3352)
        data.randomTrainingTestByPercent(np.array(quantidade_por_classe).copy(), 0.5)
        data.params = dict(kernel_type=cv2.SVM_RBF, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0, k_fold=2)
        svm = cv2.SVM()
        svm.train_auto(np.float32(data_set.atributes[data.Training_indexes]),
                       np.float32(data_set.labels[data.Training_indexes]), None, None, params=data.params)
        results = svm.predict_all(np.float32(data_set.atributes[data.Testing_indexes]))
        data.setResultsFromClassfier(results, data_set.labels[data.Testing_indexes])
        data_set.append(data)
    experiment.addDataSet(data_set,
                          "GLCM 24 attributes with {} bits {} rounds vs SVM Kernel RBF. base numbers, 50% per class. ".format(
                              file_index, it_for_data_set))
experiment.save("EXPERIMENTS/EXP01-GLCM_SVM.gzip")
