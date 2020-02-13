import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image,line_paramaters):
    slope,intercept=line_paramaters
    y1=image.shape[0] #total height of image i.e we will get bottom of image
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)  #y=mx+c --> x=(y-c)/m
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines): #to find the avg st. line(single) from the all lines
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1) # returns [[slope,intercept(y)],[],[],...]
        slope=parameters[0]
        intercept=parameters[1]
        if(slope<0):
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg=np.average(right_fit,axis=0) #axis=0 means average of all in one column wise.
    try:
        left_line=make_coordinates(image,left_fit_avg)
        right_line=make_coordinates(image,right_fit_avg)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e, '\n')
 #print error to console
        return None

def canny(img): #Calculates gradient i.e Outlines the image
                ##  MOST SHARP CHANGES IN THE PIXLES
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0) # - blur image
    canny_img=cv2.Canny(blur,50,150)
    return canny_img

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
         #converting [[x1,y1,x2,y2]] ---> [x1,y1,x2,y2] 2D to 1D
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image


def area_of_interest(image): #gives the area of road we are interested in
#We will use this mask to focus on one part of the image and mask out all other
    height=image.shape[0]
    polygons=np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_img=cv2.bitwise_and(image,mask) # ands the two images to get the area
    return masked_img #return white Outline of the one lane

'''
img=cv2.imread('test_image.jpg')
lane_img=np.copy(img)#Copies the intensity matrix of img to lane_img
canny=canny(img)
cropped_img=area_of_interest(canny)
lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
averaged_lines=average_slope_intercept(lane_img,lines)
line_image=display_lines(lane_img,averaged_lines)
final_img=cv2.addWeighted(lane_img,0.8,line_image,1,1)
cv2.imshow("result",final_img)
cv2.waitKey(0)
'''
#plt.imshow(canny)
#plt.show()
cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    canny_img=canny(frame)
    cropped_img=area_of_interest(canny_img)
    lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=average_slope_intercept(frame,lines)
    line_image=display_lines(frame,averaged_lines)
    final_img=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow("result",final_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
