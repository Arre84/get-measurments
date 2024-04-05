"""get-measures.py

    This pythons script computes the real measure of a group of selected lines and figures
    given the calibration parameters of the camera and the the depth between the camera and
    the choseen lines, figures, objects etc.

    Author:Emilio Arredondo Payán
    Organisation: Universidad de Monterrey
    Contact: Emilio.Arredondop@udem.edu
    First created: Saturday 30 March 2024
    Last updated: Friday 05 April 2024

    EXAMPLE OF USAGE:
    python .\get-measurements.py -c 1 --z 30 --cal_file .\calibration_data.json
    
"""

import cv2
import numpy as np
import argparse as arg
from numpy.typing import NDArray
import os
import json

#Global variables
points = [] #Variable to colect the points in the coordinates in the drwaing
state = False #Variable to define if we want to finish making points and estimate the perimeter.
afk = 1 #Variable to not write the result in terminal every frame.

def user_interaction()-> arg.ArgumentParser:
    """
    Parse command-line arguments for segment measurment.
    Returns:
        argparse.ArgumentParser: The argument parser object configured for segment measurment.
    """
    parser = arg.ArgumentParser(description='dimension measurment')
    parser.add_argument("-c",'--cam_index', 
                        type=int, 
                        default=0, 
                        help='Camera index for the measurment')
    parser.add_argument('--z', 
                        type=float, 
                        help='Depth within the camera and the object')
    parser.add_argument('--cal_file', 
                        type=str, 
                        help='Path to the calibration JSON object')
    args = parser.parse_args()
    return args

def load_JSON(args:arg.ArgumentParser)->NDArray:
    """
    Load camera calibration parameters from a JSON file.

    Args:
        args: Parsed command-line arguments.

    Returns:
        camera_matrix: Camera matrix.
        distortion_coefficients: Distortion coefficients.

    This function will raise a warning if the JSON file 
    does not exist and shut down the program.
    """
    # Check if JSON file exists
    json_filename = args.cal_file
    check_file = os.path.isfile(json_filename)

    # If JSON file exists, load the calibration parameters
    if check_file:
        f = open(json_filename)
        json_data = json.load(f)
        f.close()
        
        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefficients = np.array(json_data['distortion_coefficients'])
        return camera_matrix, distortion_coefficients
    
    # Otherwise, the program finishes
    else:
        print(f"The file {json_filename} does not exist!")
        exit(-1)

def undistort_images(
        img, 
        mtx:NDArray, 
        dist:NDArray, 
        )->NDArray:
        """
        Undisorts the current frame with with the calibration paramters
        Args: 
            img: Current frame or image.
            mtx: Matrix with calibration parameters.
            dist: Matrix with distorsion parameters.

        Returns: undisorted image.
        """
        # Get size
        h,  w = img.shape[:2]

        # Get optimal new camera
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Crop image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst


def initialise_camera(args:arg.ArgumentParser)->cv2.VideoCapture:
    """
    Opens the video path or camera index provided by the user.

    args:
        args: cotains the camera index to open.
    
    Returns:
        Cap: Variable with the frame.
    """
    cap = cv2.VideoCapture(args.cam_index)
    return cap 

# Define a callback function to capture mouse events
def mouse_interaction(event,x,y,flags,params)->None:
    """
    Performs the necessary commands to correctly save every selected point,
    with the left mouse button it saves the point coordinates, middle mouse button 
    connects the last point with the first point, creating a fuigure (2 lines minimum),
    ctrl key erase all points.

    Returns:
        None: This function does not return something 
    """
    global state
    global afk
    if event == cv2.EVENT_LBUTTONDOWN and state == False:
        points.append((x,y))
    if flags & cv2.EVENT_FLAG_CTRLKEY:
        points.clear()
        state = False
    if event == cv2.EVENT_MBUTTONDOWN and len(points)!= 0:
        state = True
        afk = 0
        if len(points)>2:
            points.append(points[0])

#Function to draw lines in the frame   
def draw_lines(frame:NDArray,mtx:NDArray,dist:NDArray)->None:
    """
    Function to draw the selected points with the mouse and connect those points.

    Args:
        Frame: Current frame or image.
        mtx: Matrix with calibration parameters.
        dist: Matrix with distorsion parameters.
    Retruns: 
        None: This functions does not return something 
    """

    #Variable to control the index of the matrix with the coordinates
    cnt = 0 
    #We name the window we want the drawings to be shown in
    cv2.namedWindow('image')

    #Track the mouse events
    cv2.setMouseCallback("image",mouse_interaction)

    #Loop to draw evey single dot and link these.
    for point in points:
        cv2.circle(frame,(point[0],point[1]),3,(0,0,0),-1)
        cnt +=1
        if cnt > 1:
            previous = points[cnt-2] 
            cv2.line(frame,point,previous,(0,0,0),1)

    #The frame is shown
    frame = undistort_images(frame,mtx,dist)
    cv2.imshow("image",frame) 
    return None

#Function to compute the line lenghts 
def compute_line_segments(mtx:NDArray,h:int,w:int,args:arg.ArgumentParser)->tuple[list[tuple[str, float]],list[tuple[str, float]]]:
    """
    Function to estimate the line measures projected to the real world (pixels to cm, mm, ft, etc).

    Args:
        mtx: matrix containing calibration parameters (fy,fx,cx,cy)
        h: height of the frame
        w: width of the frame 
        args: argument containing the distance between the camera and the measured object.
    Retruns:
        lengths: Estimated measure of the line segments from smaller to bigger.
        consecutive_points: Rstimated measure of the line segments in ascending order (P01-P02-P03 etc.)
    """
    cnt = 0
    global state
    lengths = []
    consecutive_points = []
    fx = mtx[0,0]*w/1600
    cx = mtx[0,2]*w/1600
    cy = mtx[1,2]*h/1200
    fy = mtx[1,1]*h/1200
    Z = args.z
    if len(points) >= 2 and state == True:
        for point in points:
            cnt +=1
            if cnt > 1:
                previous = points[cnt-2]
                x0 = (previous[0]-cx)*Z/fx
                y0 = (previous[1]-cy)*Z/fy
                x1 = (point[0]-cx)*Z/fx
                y1 = (point[1]-cy)*Z/fy
                x = (x1-x0)**2
                y = (y1-y0)**2
                lengths.append(("P{}{}:".format(cnt-2,cnt-1),np.sqrt(x+y)))
                consecutive_points.append(("P{}{}:".format(cnt-2,cnt-1),np.sqrt(x+y)))
    #We sort the array from smaller to bigger.
    
    lengths.sort(key=lambda x: x[1], reverse=False)
    return lengths, consecutive_points

#function to compute the perimeter of the figure
def compute_perimeter(lengths:NDArray)->float:
    """
    Function to calculate the perimeter of the figure
    Args:
        lengths: List containing the line lengths
    Retruns:
        perimeter: Variable with the calculated perimeter.
    """
    perimeter = 0
    #We just compute it if there are more than two lines.
    if len(lengths)>2:
        for point in lengths:
            perimeter += point[1]
    return perimeter

def print_results(lengths:list,perimeter:float,consecutive:list,mtx:NDArray,h:int,w:int,args:arg.ArgumentParser)->None:
    """
    Function to print results in the given case, just the corridnates if only one point is gives,
    the line lengths in case only two points were given, the consecutive distances between points 
    or lines, ascending order of the line segments and the perimeter. 

    Args:
        lengths: List containing the line lengths in ascending order.
        perimeter: Perimeter of the figure.
        consecutive: Lis cntaining the line lengths in consecutive order.
        mtx: Matrix containing the calibration parameters.
        h: height of the image.
        w: width of the image.
        args: argument containing the depth.
    Returns:
        None: thisfunction does not return something.
    """
    global afk #Variable to contrl if the results should be printed or it should stay the same
    fx = mtx[0,0]*w/1600
    cx = mtx[0,2]*w/1600
    cy = mtx[1,2]*h/1200
    fy = mtx[1,1]*h/1200
    Z = args.z
    print("------------------------------------")
    print("Distance between consecutive points")
    for length in consecutive:
        print(length[0],length[1],"cm")
    print("List in ascending order")
    for length in lengths:
        print(length[0],length[1],"cm")
    if perimeter!=0:
        print("Perimeter:",perimeter,"cm")
    if len(points) < 2 and state == True:
        P = points[0]
        x = (P[0]-cx)*Z/fx
        y = (P[1]-cy)*Z/fy
        print("Just one point was selected with coordinates",x,y)
    print("------------------------------------")
    afk = 1
    return None

def pipeline()->None:
    """
    Function to perform the line drawing and calculate the distances of each line and the perimeter in case it 
    is needed.
    Returns:
        None: This function does not return something
    """
    global afk
    args = user_interaction()
    cap = initialise_camera(args)
    camera_matrix,distortion_coefficients = load_JSON(args)
    while cap.isOpened():
        # Read current frame
        ret, frame = cap.read()
        
        # Check if the image was correctly captured
        if not ret:
            print("It seems like a problem has occured, try running the program again, in case the\n"
                   "problem keeps ocurring, call : 614-345-3164")
            break
        height,width,_ = frame.shape
        #Draw the mouse callback
        draw_lines(frame,camera_matrix,distortion_coefficients)
        #Estimate and sort the line lengths 
        lengths,consecutive_points = compute_line_segments(camera_matrix,height,width,args)
        #Estimate the perimeter in case the figure has more than 2 lines
        perimeter = compute_perimeter(lengths)

        if afk == 0:
            print_results(lengths,perimeter,consecutive_points,camera_matrix,height,width,args)

        key = cv2.waitKey(20)
        if key == ord('q') or key == 27:
            print("Programm finished!")
            break
    cv2.destroyAllWindows()
    cap.release()
    return None

if __name__ == "__main__":
    pipeline()