# import libraries here
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

def morph_a_lot(morph):
    
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_er = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    
    morph = cv2.dilate(morph, kernel_dil, iterations=6)
    morph = cv2.erode(morph, kernel_er, iterations=19)

    return morph

def morph_a_little(morph):
   
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    kernel_er = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))

    morph = cv2.dilate(morph, kernel_dil, iterations=2)
    morph = cv2.erode(morph, kernel_er, iterations=4)
    morph = cv2.dilate(morph, kernel_dil, iterations=1)
    morph = cv2.erode(morph, kernel_er, iterations=1)
    morph = cv2.dilate(morph, kernel_dil, iterations=1)

    return morph


def count_cars(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj prebrojanih automobila. Koristiti ovu putanju koja vec dolazi
    kroz argument procedure i ne hardkodirati nove putanje u kodu.

    Ova procedura se poziva automatski iz main procedure i taj deo koda nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih automobila
    """

    # TODO - Prebrojati auta i vratiti njihov broj kao povratnu vrednost ove procedure
    
    car_contours = []

    img = cv2.imread(image_path)
    
    if img.shape[0] * img.shape[1] > 900 * 650:
        min_wh = 5 #minimalna visina/sirina za konturu auta
        max_wh = 500 #maximalna visina/sirina za konturu auta
        min_line = 8 #minimalna duzina linije iz LSD-a

        line_width = 10 #sirina linija koje se brisu 
        blur = 1 #da li se radi blur

        low_threshold = 80 #donja canny granica
        high_threshold = 120 #gornja canny granica

        morph_lvl = 1 #nivo morfoloskih operacija nakon brisanja linija

    else:
        min_wh = 10
        max_wh = 500
        min_line = 15

        line_width = 5
        blur = 0

        low_threshold = 150
        high_threshold = 200

        morph_lvl = 0
        
    
    img = cv2.resize(img, (420, 280), cv2.INTER_LINEAR)
     
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #https://docs.opencv.org/3.4/d4/d13/tutorial_py_filtering.html
    if(blur == 1):
        blur_gray = cv2.GaussianBlur(gray, (7, 7), 0)
    else: 
        blur_gray = np.copy(gray)
    
    #https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    canny = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #https://docs.opencv.org/3.4/df/dfa/tutorial_line_descriptor_main.html
    lsd = cv2.createLineSegmentDetector(0)
    lines_all = lsd.detect(canny)[0]

    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    
    morph = np.copy(canny) #kopija ivica posle cannija, cuva se canny zbog prikaza i namestanja canny parametara

    if morph_lvl == 1:
        morph = morph_a_lot(morph)
    else:
        morph = morph_a_little(morph)
    
    for line in lines_all:
        for x1,y1,x2,y2 in line: 
            if abs(x2-x1) > min_line or abs(y2-y1) > min_line:
                cv2.line(morph, (x1, y1), (x2,y2), (0, 0, 0), line_width) #brise prave linije sa canny 
                cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 1) #cv2.line iscrtava liniju na line_image

    _, contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours: # za svaku konturu
        _, size, _ = cv2.minAreaRect(contour) # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if width > min_wh and width < max_wh and height > min_wh and height < max_wh: # uslov da kontura pripada kolima
            car_contours.append(contour) # ova kontura pripada kolima

    cv2.drawContours(img, car_contours, -1, (255, 0, 0), 1)

    f, axarr = plt.subplots(2,2)
    axarr[0, 0].imshow(morph, 'gray')
    axarr[0, 1].imshow(canny, 'gray')
    axarr[1, 0].imshow(line_image, 'gray')
    axarr[1, 1].imshow(img)
    #plt.show()

    car_count = len(car_contours)
    return car_count

