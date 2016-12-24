import numpy as np
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
	true_video = cv2.VideoCapture('kovrov_val.avi')
	res_video = cv2.VideoCapture('test.avi')

	stat = {'re':  [],
	        'pr':  [],
	        'sp':  [],
	        'fpr': [],
	        'fnr': [],
	        'pwc': [],
	        'F-m': []}

	names = {'re':  'Recall',
	         'pr':  'Precision',
	         'sp':  'Specificity',
	         'fpr': 'False positive Rate',
	         'fnr': 'False negative rate',
	         'pwc': 'Percentage of wrong Classifications',
	         'F-m': 'F-measure'}

	while True:
		ret_true, original = true_video.read()
		ret_res, result = res_video.read()

		if ret_true == False or ret_res == False:
			break

		original = original[:, :, 0]
		result = result[:, :, 0]

		thr = 100
		original[original < thr] = 0
		original[original >= thr] = 1

		result[result < thr] = 0
		result[result >= thr] = 1

		TP = np.sum((result[original == 1]) == 1)
		FP = np.sum((result[original == 0]) == 1)
		FN = np.sum((result[original == 1]) == 0)
		TN = np.sum((result[original == 0]) == 0)

		Re = TP / (TP + FN)
		Pr = TP / (TP + FP)
		Sp = TN / (TN + FP)
		FPR =  FP / (TN + FP)
		FNR = FN / (TP + FN)
		PWC = 100 * (FN + FP) / (TP + FN + FP + TN)
		F = 2 * (Pr * Re) / (Pr + Re)

		stat['re'].append(Re)
		stat['pr'].append(Pr)
		stat['sp'].append(Sp)
		stat['fpr'].append(FPR)
		stat['fnr'].append(FNR)
		stat['pwc'].append(PWC)
		stat['F-m'].append(F)

	for key in stat.keys():
		plt.figure()
		plt.plot(stat[key])
		plt.grid(True)
		plt.ylabel(names[key])
		plt.xlabel('Itr')
		plt.title(names[key])
	plt.show()
