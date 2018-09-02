import cv2
def extract(image_path,type="ORB",ifShow=False):
    image=cv2.imread(image_path,0)
    if type=="AKAZE":
        fe = cv2.AKAZE_create()
    elif type=="BRISK":
        fe = cv2.BRISK_create()
    elif type=="KAZE":
        fe = cv2.KAZE_create()
    elif type=="ORB":
        fe = cv2.ORB_create()
    elif type =="SIFT":
        fe = cv2.xfeatures2d.SIFT_create()
    elif type =="SURF":
        fe = cv2.xfeatures2d.SURF_create()
    kp, des = fe.detectAndCompute(image, None)
    if ifShow:
        im2show=cv2.drawKeypoints(image,kp,None,color=(0,255,0),flags=0)
        print('%s descriptors number is %d .' % (type,len(des)))
        cv2.imshow(type,im2show)
        cv2.waitKey(0)
    return des
def test():
    extract("lena.png", type="ORB",ifShow=True)
    extract("lena.png", type="SIFT", ifShow=True)
    extract("lena.png", type="SURF", ifShow=True)
    extract("lena.png", type="AKAZE", ifShow=True)
    extract("lena.png", type="BRISK", ifShow=True)
    extract("lena.png", type="KAZE", ifShow=True)