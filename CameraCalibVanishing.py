from cv2 import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math

def click_event(event,x,y,flags,params):
    results=[]
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        results.append((x,y))
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', image)
    

# reading image and creating a border around the image
image=cv2.imread(r'C:\Users\kc\Desktop\.venv\Perception\Q2\IMG_3606.jpg')
print("shape of the image")
print(image.shape)
borderoutput = cv2.copyMakeBorder(
    image, 150, 150, 150, 150, cv2.BORDER_CONSTANT, value=[0, 0, 0])
cv2.imwrite('output.png', borderoutput)



cv2.imshow('image',image)
cv2.setMouseCallback('image',click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()


# calculating location of vanishing point in x direction
m1=(125-232)/(468-131)
print("the value of m1 is")
print(m1)
c1=125-m1*468
print("the value of c1 is")
print(c1)
m2=(204-338)/(452-144)
print("the value of m2 is")
print(m2)
c2=204-m2*452
print("the value of c2 is")
print(c2)
# find location of first vanishing point by equating two equations
x_vanishing_1=(c2-c1)/(m1-m2)
y_vanishing_1=m1*x_vanishing_1+c1
print(f'location of vanishing point in x direction is {(x_vanishing_1,y_vanishing_1)}')

# location of vanishing point using least square (x-direction)
x1 = np.array([215, 226])
y1=np.array([317,288])
A_1 = np.vstack([x1, np.ones(len(x1))]).T
m1_least, c1_least = np.linalg.lstsq(A_1, y1, rcond=None)[0]
print('m-least square is')
print(m1_least)
print('c-least square is')
print(c1_least)

x2 = np.array([108,114])
y2=np.array([171,144])
A_2 = np.vstack([x2, np.ones(len(x2))]).T
m2_least, c2_least = np.linalg.lstsq(A_2, y2, rcond=None)[0]
print('m-least square is')
print(m2_least)
print('c-least square is')
print(c2_least)
print("the location of first vanishing point using least square method")
x0_least=((c2_least-c1_least)/(m1_least-m2_least))
y0_least=(m1_least*x0_least+c1_least)
print(x0_least,y0_least)

START_POINT_1=(215,317)
END_POINT_1=(-121,1204)
COLOR_1=(0,255,0)
THICKNESS_1=2
cv2.line(image,START_POINT_1,END_POINT_1,COLOR_1,THICKNESS_1)
cv2.imshow("drawing_line",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

START_POINT_2=(108,171)
END_POINT_2=(-121,1204)
COLOR_2=(0,255,0)
THICKNESS_2=2
cv2.line(image,START_POINT_2,END_POINT_2,COLOR_2,THICKNESS_2)
cv2.imshow("drawing_line",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# location of vanishing point in y-direction
m3=(138-230)/(21-122)
c3=138-m3*21
m4=(221-330)/(34-139)
c4=221-m4*34
x_vanishing_2=(c4-c3)/(m3-m4)
y_vanishing_2=m3*x_vanishing_1+c3
print(f'location of vanishing point in y direction is {(x_vanishing_2,y_vanishing_2)}')


# location of vanishing point using least square (y-direction)
x3 = np.array([351,328,298,274,254,237])
y3=np.array([246,254,265,273,282,286])
A_3 = np.vstack([x3, np.ones(len(x3))]).T
m3_least, c3_least = np.linalg.lstsq(A_3, y3, rcond=None)[0]
print('m-least square is')
print(m3_least)
print('c-least square is')
print(c3_least)

x4=np.array([242,215,172,154,138,121])
y4=np.array([132,136,134,136,138,141])
A_4 = np.vstack([x4, np.ones(len(x4))]).T
m4_least, c4_least = np.linalg.lstsq(A_4, y4, rcond=None)[0]
print('m-least square is')
print(m4_least)
print('c-least square is')
print(c4_least)

print("the location of first vanishing point in y direction using least square method")
x1_least=((c4_least-c3_least)/(m3_least-m4_least))
y1_least=(m3_least*x0_least+c3_least)
print(x1_least,y1_least)

START_POINT_3=(350,245)
END_POINT_3=(-2172,413)
COLOR_4=(255,0,0)
THICKNESS_4=2
cv2.line(image,START_POINT_3,END_POINT_3,COLOR_4,THICKNESS_4)
cv2.imshow("drawing_line",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
START_POINT_4=(267,301)
END_POINT_4=(-2172,413)
COLOR_4=(255,0,0)
THICKNESS_4=2
cv2.line(image,START_POINT_4,END_POINT_4,COLOR_4,THICKNESS_4)
cv2.imshow("drawing_line",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# third vanishing direction
m5=(248-285)/(170-203)
c5=248-m5*170
m6=(210-244)/(172-215)
c6=210-m6*172
x_vanishing_3=(c6-c5)/(m5-m6)
y_vanishing_3=m5*x_vanishing_1+c5
print(f'location of vanishing point in z direction is {(x_vanishing_3,y_vanishing_3)}')



# location of vanishing point using least square (z-direction)
x5 = np.array([226, 190,161,138,114])
y5=np.array([292,244,205,175,146])
A_5 = np.vstack([x5, np.ones(len(x5))]).T
m5_least, c5_least = np.linalg.lstsq(A_5, y5, rcond=None)[0]
print('m-least square is')
print(m5_least)
print('c-least square is')
print(c5_least)


x6=np.array([210,184,160,138,108])
y6 = np.array([314,275,242,209,170])
A_6 = np.vstack([x6, np.ones(len(x6))]).T
m6_least, c6_least = np.linalg.lstsq(A_6, y6, rcond=None)[0]
print('m-least square is')
print(m6_least)
print('c-least square is')
print(c6_least)

print("the location of first vanishing point in z direction using least square method")
x2_least=((c6_least-c5_least)/(m5_least-m6_least))
y2_least=(m5_least*x0_least+c5_least)
print(x2_least,y2_least)

START_POINT_5=(226,292)
END_POINT_5=(-194,-164)
COLOR_6=(0,0,255)
THICKNESS_6=2
cv2.line(image,START_POINT_5,END_POINT_5,COLOR_6,THICKNESS_6)
cv2.imshow("drawing_line",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
START_POINT_6=(210,314)
END_POINT_6=(-194,-164)
COLOR_6=(0,0,255)
THICKNESS_6=2
cv2.line(image,START_POINT_6,END_POINT_6,COLOR_6,THICKNESS_6)
cv2.imshow("drawing_line",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# vanishing point v2
cv2.imshow('board',borderoutput)
cv2.setMouseCallback('board',click_event)
cv2.line(borderoutput,(377,438),(850,269),(0,255,0),2)
cv2.imshow("board",borderoutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.line(borderoutput,(374,466),(850,269),(0,255,0),2)
cv2.imshow("board",borderoutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_text = cv2.putText(borderoutput, 'V2', (850,269), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 3, cv2.LINE_AA)
cv2.imshow('image_text',image_text)
cv2.waitKey(0)
cv2.destroyAllWindows()


# vanishing point v3
cv2.imshow('board',borderoutput)
cv2.setMouseCallback('board',click_event)
#cv2.line(borderoutput,(363,466),(165,184),(0,255,0),2)
cv2.line(borderoutput,(363,466),(80,99),(0,255,0),2)
cv2.imshow("board",borderoutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.line(borderoutput,(376,437),(165,184),(0,255,0),2)
cv2.line(borderoutput,(376,437),(80,99),(0,255,0),2)
cv2.imshow("board",borderoutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_text = cv2.putText(borderoutput, 'V3', (80,99), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 3, cv2.LINE_AA)
cv2.imshow('image_text',image_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

# vanishing point v1
cv2.imshow('board',borderoutput)
cv2.setMouseCallback('board',click_event)
#cv2.line(borderoutput,(374,443),(274,687),(0,255,0),2)
cv2.line(borderoutput,(374,443),(69,738),(0,255,0),2)
cv2.imshow("board",borderoutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.line(borderoutput,(263,298),(274,687),(0,255,0),2)
cv2.line(borderoutput,(263,298),(69,738),(0,255,0),2)
cv2.imshow("board",borderoutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_text = cv2.putText(borderoutput, 'V1', (69,738), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 3, cv2.LINE_AA)
cv2.imshow('image_text',image_text)
cv2.waitKey(0)
cv2.destroyAllWindows()


def cameraCalibration(vanishing_point_x,vanishing_point_y,vanishing_point_z):
    '''
    Input:
        Vanishing_point_x: tuple (x,y) for vanishing point in x_direction
        Vanishing_point_y: tuple (x,y) for vanishing point in y_direction
        Vanishing_point_z: tuple (x,y) for vanishing point in z_direction
    Output:
        Principal point: (x0,y0)
        focal length: f
    '''

    # form A matrix A (2,2)
    A= np.array([[vanishing_point_x[0]-vanishing_point_z[0], vanishing_point_x[1]-vanishing_point_z[1]], 
                [vanishing_point_y[0]-vanishing_point_z[0],vanishing_point_y[1]-vanishing_point_z[1]]])
    print(A.shape)
    
    # form matrix B
    b=np.array([[vanishing_point_y[0]*(vanishing_point_x[0]-vanishing_point_z[0])+vanishing_point_y[1]*(vanishing_point_x[1]-vanishing_point_z[1])],
                [vanishing_point_x[0]*(vanishing_point_y[0]-vanishing_point_z[0])+vanishing_point_x[1]*(vanishing_point_y[1]-vanishing_point_z[1])]])
    print(b.shape)

    # solve for x (contains principal point coordinates)
    part1=np.linalg.inv(np.matmul(A.T,A))
    part2=np.matmul(A.T,b)
    x=np.matmul(part1,part2)
    print(f"The solution is {x}")
    x0=x[0,:]
    y0=x[1,:]

    # now compute the focal length
    focal_length=np.sqrt(-(vanishing_point_x[0]-x0)*(vanishing_point_y[0]-x0)-(vanishing_point_x[1]-y0)*(vanishing_point_y[1]-y0))

    return (x0,y0),focal_length

(princ_x,princ_y),focal_len=cameraCalibration((x0_least,y0_least),(x1_least,y1_least),(x2_least,y2_least))
print(f'the x principal point is{princ_x}')
print(f'the y principal point is{princ_y}')
print(f'the focal lengrg is{focal_len}')

































