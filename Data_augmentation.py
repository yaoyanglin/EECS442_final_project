import numpy as np
import cv2
import os

# # Load the Data
def load_images(path):
    folder_names = [d for d in os.listdir('./English/Fnt/') if d.startswith('Sample')]
    for folder in folder_names:
        print("reading file names in " + folder)
        print(path + folder + "/")
        names = [d for d in os.listdir(path + folder + "/") if d.endswith('.png')]
        #for name in names:
        for name in names:
            img = cv2.imread(path + folder + "/"  + name, 0)

            pts1 = np.float32([[10,10],[118,10],[10,118],[118,118]])
            pts2 = np.float32([[0,0],[108,20],[0,128],[108,108]])
            img = cv2.bitwise_not(img)

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (128, 128))
            dst = cv2.bitwise_not(dst)
            cv2.imwrite(path + folder + "/" + name[:-5] + "_affine1.png", dst)

            # another projection
            pts1 = np.float32([[10, 10], [118, 10], [10, 118], [118, 118]])
            pts2 = np.float32([[20, 20], [128, 0], [20, 108], [128, 128]])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (128, 128))
            dst = cv2.bitwise_not(dst)
            cv2.imwrite(path + folder + "/" + name[:-5] + "_affine2.png", dst)

            # another projection
            pts1 = np.float32([[10, 10], [118, 10], [10, 118], [118, 118]])
            pts2 = np.float32([[20, 20], [108, 20], [0, 128], [128, 128]])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (128, 128))
            dst = cv2.bitwise_not(dst)
            cv2.imwrite(path + folder + "/" + name[:-5] + "./affine3.png", dst)

            # another projection
            pts1 = np.float32([[10, 10], [118, 10], [10, 118], [118, 118]])
            pts2 = np.float32([[0, 0], [128, 0], [20, 108], [108, 108]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (128, 128))
            dst = cv2.bitwise_not(dst)
            cv2.imwrite(path + folder + "/" + name[:-5] + "./affine4.png", dst)

            # another rotation
            M = cv2.getRotationMatrix2D((64, 64), 10, 1)
            dst = cv2.warpAffine(img, M, (128, 128))
            dst = cv2.bitwise_not(dst)
            cv2.imwrite(path + folder + "/" + name[:-5] + "./rotation1.png", dst)

            # another rotation
            M = cv2.getRotationMatrix2D((64, 64), -10, 1)
            dst = cv2.warpAffine(img, M, (128, 128))
            dst = cv2.bitwise_not(dst)
            cv2.imwrite(path + folder + "/" + name[:-5] + "./rotation2.png", dst)



load_images('./English/Fnt/')







