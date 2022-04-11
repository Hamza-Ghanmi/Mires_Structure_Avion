import cv2


img = cv2.imread("Sequence_000000.bmp")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thrG = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
MR = cv2.MSER_create()
regions, _ = MR.detectRegions(thrG)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
imgContours = img.copy()
centers = []
radius = []
n = 0
for i in range(len(hulls)):
    peri = cv2.arcLength(hulls[i], True)
    approx = cv2.approxPolyDP(hulls[i], 0.01 * peri, True)
    objCor = len(approx)
    area = cv2.contourArea(hulls[i])
    if objCor > 7 and 410 < area < 790:
        cnt, rad = cv2.minEnclosingCircle(hulls[i])
        exist = False
        for j in range(len(centers)):
            if (centers[j][0]-3 < cnt[0] < centers[j][0]+3) and (centers[j][1]-3 < cnt[1] < centers[j][1]+3):
                exist = True
        if not exist:
            n = n+1
            cv2.drawContours(imgContours, hulls, i, (0, 255, 0))
            cv2.putText(imgContours, str(n),
                        (round(cnt[0]), round(cnt[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 0, 255), 1)
            centers.append(cnt)
            radius.append(rad)

h = int(img.shape[0]/3)
w = int(img.shape[1]/3)
imgResize = cv2.resize(imgContours, (w, h))
thrGResize = cv2.resize(thrG, (w, h))
cv2.imwrite("imgContours.bmp", imgContours)
cv2.imshow("imgContours", imgResize)
cv2.imshow("imgThresh", thrGResize)
cv2.waitKey()