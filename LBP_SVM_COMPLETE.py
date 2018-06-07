import cv2

from MachineLearn.Classes.data import Data
from MachineLearn.Classes.Extractors.LBP import Lbp8Bits
import numpy as np

from MachineLearn.Classes.data_set import DataSet
from MachineLearn.Classes.experiment import Experiment

#
# data_set = np.loadtxt("DATABASE.txt", delimiter=" ")
# data_set[data_set == 1] = 255
# att_dataset = []
# for i in data_set:
#     olbp = Lbp8Bits(np.reshape(i[:-1], (35, 35)))
#     olbp.calculate_attributes()
#     att_dataset.append(olbp.export_to_classifier("CLASS-" + str(i[-1])))
#     np.savetxt("LBP_8B.txt", np.array(att_dataset), delimiter=",", fmt="%s")
#

experiment = Experiment()
it_for_data_set = 50
for file_index in [4,8]:
    print "DATASET: ", file_index
    atts = np.loadtxt("LBP_{}B.txt".format(file_index), delimiter=",", usecols=(x for x in range(2**file_index)))
    labels = np.loadtxt("LBP_{}B.txt".format(file_index), delimiter=",", usecols=(-1), dtype=object)
    labels[labels == 'CLASS-255.0'] = 'CLASS-1.0'

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
        data.params = dict(kernel_type=cv2.SVM_SIGMOID, svm_type=cv2.SVM_C_SVC, gamma=2.0, nu=0.0, p=0.0, coef0=0, k_fold=2)
        svm = cv2.SVM()
        svm.train_auto(np.float32(data_set.atributes[data.Training_indexes]),
                       np.float32(data_set.labels[data.Training_indexes]), None, None, params=data.params)
        results = svm.predict_all(np.float32(data_set.atributes[data.Testing_indexes]))
        data.setResultsFromClassfier(results, data_set.labels[data.Testing_indexes])
        data_set.append(data)
    experiment.addDataSet(data_set,
                          "LBP attributes with {} bits {} rounds vs SVM Kernel SIGMOID. base numbers, 50% per class. ".format(
                              file_index, it_for_data_set))
experiment.save("EXPERIMENTS/EXP02-LBP_SVM.gzip")
