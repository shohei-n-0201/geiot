import os
import math
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector()

shirtFolderPath = "Resources/Glasses"
listShirts = os.listdir(shirtFolderPath)
# print(listShirts)
fixedRatio = 320 / 190  # widthOfShirt/widthOfPoint11to12
shirtRatioHeightWidth = 150 / 500
imageNumber = 0
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findPose(img, draw=False)
    # img = cv2.flip(img,1)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        # center = bboxInfo["center"]
        lm11 = lmList[3]  # 右目外側
        lm12 = lmList[6]  # 左目外側
        print(lm11, lm12)
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        #print(widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)
    
        # 傾き計算
        angle = math.degrees(math.atan2(lm11[1] - lm12[1], lm11[0] - lm12[0]))

        # 画像の回転
        M = cv2.getRotationMatrix2D((widthOfShirt // 2, int(widthOfShirt * shirtRatioHeightWidth) // 2), -angle, 1)
        imgShirt = cv2.warpAffine(imgShirt, M, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))

        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except:
            pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)