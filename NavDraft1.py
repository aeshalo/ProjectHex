# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 08:47:50 2017

@author: Alex Johnson

This program performs automated missions by instructing an attached microcontroller.
This code also has Computer vision elements, which require a camera.
The Microcontroller is communicated with through the dronekit api and opencv is
used to talk to the camera and perform CV. Tesseract is used for OCR
all used libraries must be place in this folder or in a folder specified by 
adding it to the path like below.

This code is structured as follows: Each small task, like going to a waypoint 
or arming the drone, is coded into it's own 'task' subcalss. The Parser fucntion
takes a specified text file in the correct format and generates a list of task
instances, each with it's specified arguments. This list of task objects is then
given to a stack handeller, that executes the tasks one after another.

Camera tasks are actually merely signals to the camera loop, operating in a seperate
thread so the potentially computationally intensive computer vision code does not
interrupt the mission critical nav code.

All tasks return zero if successful. If failed they can return anyting else or 
not return at all. a failed task results in a failed mission, and the drone will
be put in 'alt_hold' mode so that the pilot can intervene.
"""
#===================LIBRARIES,GLOBAL VARIABLES=====================
import sys
sys.path.insert(0,'C:\Program Files\Anaconda3\Lib\site-packages')
from dronekit import connect, VehicleMode, LocationGlobalRelative
from pymavlink import mavutil
import time
import math
import threading
import numpy as np
import cv2
import subprocess
testphase=False
bw=False
Stack = None
StackPosition = None
vehicle = None
CameraOnline = None
#=================STANDALONE FUNCTIONS=======================
def metersToDegLat(meters): #wanna go five meters north? here's that in deg lattitude
    return (math.fabs(((180/math.pi)*meters)/6378000))
    
def metersToDegLon(meters,lat):#like above but for longditude. this depends on lat
    return (math.fabs((((180/math.pi)*meters)/6378000)*math.cos(lat)))
    
#=================CAMERA STUFF===================================
    
#CALCULATE THE LINEAR DISTANCE BETWEEN TWO POINTS IN 2D
def dist(a,b):   
    return np.sqrt(np.sum((a-b)**2))
    
#CALCULATE THE DISTANCE OF THE SQUARE IN THE Z AXIS FROM THE POSITIONS OF THE 
#CORNERS (INCOMPLETE - just returns size of square in image)
def zdist(corners):
    #incomplete
    #calculate length of one side
    sidelength=cv2.arcLength(corners,True)/4
    return int(sidelength)    
    
#CHECKS IF SHAPE IS SQUARE    
def SquareTest(corners):
    #checks for four sides
    if corners.shape[0]!=4:
        return False
    #checks if sides are the same length and diagonals are same length
    aspectratio1=(dist((corners[1,0]),corners[2,0]))/(dist((corners[2,0]),corners[3,0]))
    aspectratio2=(dist(corners[0,0],corners[2,0]))/(dist(corners[1,0],corners[3,0]))
    if (aspectratio1>=0.7) and (aspectratio1<=1.3) and (aspectratio2>=0.7) and (aspectratio2<=1.3):
        return True
    else: return False

#FIND THE X AND Y POSITIONS OF THE SQUARE FROM THE POSITION OF THE SQUARE IN THE IMAGE
#(NEEDS TO BE ALTERED FOR CAMERA RESOLUTION)
def xandy(Centre):
    xvalue=Centre[0,0,0]-320
    yvalue=Centre[0,0,1]-240
    return xvalue, yvalue

#SMOOTH OUT DATA, INCLUDING PREDICTING
def positionsmoother(currentposition,prevposition,predictedposition,loopcounter):
    #conditions for square not detected
    if None in currentposition:
        currentposition=[None,None,None]
    #if square found twice consecutively
    elif not(None in prevposition):
        #if the square moves by more than the tolerance in that direction, it doesn't count it
        #(z,y,x)... don't ask why
        tolerances=[50,300,400]
        for i in range(3):
            if np.abs(float(currentposition[i])-float(prevposition[i]))>=tolerances[i]:
                currentposition=[None,None,None]
                break
    #fill in the gaps in the data
    positionminus1=list(predictedposition[:,1])
    #if no square is detected, assume it is at the previous predicted value
    #if this is done more than 10 times in a row, assume it is no longer there
    if currentposition[0]==None:
        if loopcounter<=10:
            #predictedposition has a 4th row, with either 0 or 1 for if the square has been detected or not
            positionminus1[3]=0
            newcolumn=np.array(positionminus1)
            loopcounter+=1
        else:
            newcolumn=np.array([None,None,None,0])
    else:
        loopcounter=0
        currentposition.append(1)
        newcolumn=np.array(currentposition)
    #update position matrix
    predictedposition[0:4,0:2]=predictedposition[0:4,1:3]
    predictedposition[0:4,2]=newcolumn
    return predictedposition, loopcounter
    
#RETURN A CROPPED IMAGE, THE SQUARE DETECTED 
def crop2square(mask,img,Centre,corners,bw):
    #Calculate the angle of square by taking the arctan of the positions of two of the corners
    a=(float(corners[0,0,0])-float(corners[1,0,0]))
    b=(float(corners[0,0,1]-float(corners[1,0,1])))
    #protect against dividing by zero, else calculate tangent
    if b==0: angle=90
    else: angle=(180/np.pi)*np.arctan(a/b)
    #prepare for rotation
    rows,cols,depth = mask.shape
    x=Centre[0,0,0]
    y=Centre[0,0,1]
    halfsidelength=int(0.5*dist(corners[0,0],corners[1,0]))
    M = cv2.getRotationMatrix2D((x,y),0-angle,1)
    #rotate
    rotatedmask = cv2.warpAffine(mask,M,(rows,cols))
    #crop
    croppedmask=rotatedmask[(y-halfsidelength):(y+halfsidelength),(x-halfsidelength):(x+halfsidelength)]
    if bw:
        #return grayscale image of the same region for OCR
        bwimg=cv2.cvtColor( img, cv2.COLOR_BGR2GRAY );
        rotatedbwimg = cv2.warpAffine(bwimg,M,(rows,cols))
        croppedbwimg=rotatedbwimg[(y-halfsidelength):(y+halfsidelength),(x-halfsidelength):(x+halfsidelength)]
    else: croppedbwimg=croppedmask
    #cv2.imwrite("test.jpg", croppedmask)
    #cv2.imwrite("test2.jpg", mask)
    return croppedmask,croppedbwimg,bw
    
#CHECK TO SEE IF A CHARACTER FOUND BY OCR IS VALID
def charactertest(outputstring):
#make sure it's just one character
    if (len(outputstring)!=1):
        return False
    charval=ord(outputstring)
#if capital letter
    if charval<=90 and charval>=65:
        return True
#if lower case
    elif charval<=122 and charval>=97:
        return True
#if number
    if charval<=57 and charval>=48:
        return True
    else: return False

#ROTATE ROI FOR OCR
#note that the reason that the whole square is rotated and the character found
#every time instead of just that ROI is that I couldn't get the rectangular ROI
#to rotate cleanly. I kept losing important bits at the edge
def rotatecharimage(croppedmask,croppedbwimg,bw,rotation):
    #find dimensions of image
    if len(croppedmask.shape)==3:
        rows,cols,depth=croppedmask.shape
    else: rows,cols=croppedmask.shape
    #Rotate
    M = cv2.getRotationMatrix2D((int(rows/2),int(cols/2)),90*rotation,1)
    croppedmask = cv2.warpAffine(croppedmask,M,(cols,rows))
    croppedbwimg = cv2.warpAffine(croppedbwimg,M,(cols,rows))
    #find contours
    ret,thresh = cv2.threshold(cv2.cvtColor(croppedmask,cv2.COLOR_BGR2GRAY),127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sortedcont = sorted(contours, key= lambda cont: cv2.arcLength(cont,False),reverse = True)
    #Find region around character
    try:
        x,y,w,h = cv2.boundingRect(sortedcont[1])
        #crop grayscale image to ROI for OCR
        roi=croppedbwimg[(y-int(0.2*h)):(y+int(h*1.2)),(x-int(0.2*w)):(x+int(w*1.2))]
        #invert image (not hugely necessary)
        if bw: roi=255-roi
    except IndexError:
        print "Something went wrong - reinitialising"
        return None
    return roi
    
#PERFORM OCR
def tesseract(croppedmask,croppedbwimg,bw):
    #Program to classify the character.
    #tesseract OCR must be in your workspace
    #check all four possible orientations of character
    character=[]
    for rotation in range(4):
        #rotate and crop
        roi=rotatecharimage(croppedmask,croppedbwimg,bw,rotation)
        if roi==None: return []
        #here for test purposes
        if testphase:
            if rotation==0:
                cv2.imwrite('test1.jpg',roi)
            elif rotation==1:
                cv2.imwrite('test2.jpg',roi)
            elif rotation==2:
                cv2.imwrite('test3.jpg',roi)
            else:
                cv2.imwrite('test4.jpg',roi)
        #save image so it can be accessed by tesseract
        cv2.imwrite("ROIimage.jpg", roi)
        #do ocr
        subprocess.call('tesseract.exe ROIimage.jpg -psm 10 out1', True)
        #get ocr results
        fvar = open('out1.txt')
        outputstring = fvar.readline(1)
        fvar.close()
        #get rid of white space in ocr software
        outputstring=(outputstring.replace(" ","")).replace("/n","")
        if testphase: print outputstring
        #If character is valid, add to list of potential characters
        if charactertest(outputstring): character.append(outputstring)
    return character
    
#FIND SQUARE IN IMAGE
def SquareFinder(cam,img,predictedposition,currentposition,loopcounter,slower,testphase):
#CREATE MASK
    #Convert frame from bgr to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #initialise variables
    corners=[]
    squarecont=[]
    Centre=[]
    z=None
    xvalue=None
    yvalue=None
    #create the two masks
    lower_red = np.array([120,slower,0])
    upper_red = np.array([179,255,255])
    #range 2
    lower_red1 = np.array([0,slower,0])
    upper_red1 = np.array([20,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #combine the two masks into one
    myMask = cv2.bitwise_or(mask,mask1)
#FIND CONTOURS
    ret,thresh = cv2.threshold(myMask,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #Changes mask to a coloured image so that the contours can be coloured
    myMask=cv2.cvtColor(myMask, cv2.COLOR_GRAY2BGR)
    #Sorts contours by length
    sortedcont = sorted(contours, key= lambda cont: cv2.arcLength(cont,False),reverse = True)
#FINDS SQUARE CONTOURS
    for i in range(len(sortedcont)):
        #checks to see if the length of the contour is appropriate
        if cv2.arcLength(sortedcont[i],False) < 70:
            break
        #checks to see if shape formed by the contour is approximately four-sided
        corners = cv2.approxPolyDP(sortedcont[i],0.11*cv2.arcLength(sortedcont[i],True),True)
        if SquareTest(corners):
            squarecont=sortedcont[i]
                #finds centre of corners
            x=0
            y=0
            for j in range(4):
                x=x+corners[(j,0,0)]
                y=y+corners[(j,0,1)]
            n=(len(corners))
            x=x/n
            y=y/n
                #turns x and y into coordinates
            Centre = np.array([[[x,y]]])
            z=zdist(corners)
            xvalue, yvalue=xandy(Centre)
            break
#COLLECT AND DISPLAY DATA
    prevposition=currentposition
    currentposition=[xvalue,yvalue,z]
    predictedposition,loopcounter=positionsmoother(currentposition,prevposition,predictedposition,loopcounter)
#stop if k key pressed and, if square is detected 
    if testphase:
        if cv2.waitKey(1) & 0xFF == ord('k'):
            if predictedposition[3,2]==1:
                croppedmask=crop2square(myMask,Centre,corners)
                cv2.destroyAllWindows()
                cv2.imshow("Cropped Square",croppedmask)
                cv2.imwrite( "ROIimage.jpg", croppedmask )
                print 'Square positions over last three frames:'
                print(predictedposition)
                print 'Press any key to close'
                cam.release()
                cv2.waitKey(0)
            else: cam.release()
        cv2.putText(myMask,str(predictedposition),(45,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
        if predictedposition[3,2]==1:
            cv2.drawContours(myMask, squarecont, -1, (255,0,0), 3)
            cv2.drawContours(myMask, Centre, -1, (0,0,255), 5)
            cv2.drawContours(myMask, corners, -1, (0,255,0), 5)
        cv2.imshow("Video Feed",myMask)
    return predictedposition, currentposition, loopcounter, myMask,Centre,corners
    

    
    
    
    
    
    
    
def CameraLoop():
    
    #INITIALISE VARIABLES
    predictedposition=np.array([[None,None,None],[None,None,None],[None,None,None],[0,0,0]])
    loopcounter=0
    loopcounter2=0
    camloopcounter=0
    currentposition=[None,None,None]
    character=[]
    #SET LIMITS OF REDNESS TO MASK
    #text_file = open("Saturation_lower_value.txt", "r")
    #slower=int(text_file.read())
    slower=96
    #BEGIN CAPTURING VIDEO
    cam = cv2.VideoCapture(0)
    print("To close the programme, click abort and then run 'cam.release()' to re-initialise the camera")
    bw=False

    if(CameraOnline == True):
        #----------------------------------THE MAIN LOOP-------------------------------

        while(cam.isOpened()):
            #read camera
            s, img = cam.read()
            if s == False:
                print("The camera was already open - reinitialising")
                cam.release()
                camloopcounter=camloopcounter+1
                if camloopcounter==1:
                    cam = cv2.VideoCapture(0)
                    continue
                else: break
            #find square in image
            predictedposition,currentposition,loopcounter, myMask,Centre,corners=SquareFinder(cam,img,predictedposition,currentposition,loopcounter,slower,testphase)
    #add one to loop counter if square consecutively found
    if (predictedposition[3,2]==1): loopcounter2=loopcounter2+1
    else: 
        loopcounter2=0
        character=[]
    #if square found 10 times or more consecutively and character unknown
    if (loopcounter2>=10 and character==[]):
        #do ocr
        croppedmask,croppedbwimg,bw=crop2square(myMask,img,Centre,corners,bw)
        character=tesseract(croppedmask,croppedbwimg,bw)
        print character
    else:
        time.sleep(1)
        
    
#===================THE BASIC TASK OBJECT=====================
class Task(object): #all subclasses inherit from this basic object
    
    def __init__(self, name, stackpos):
        self.name = name
        self.stackpos = stackpos #the stackposition is this task's place in the stack
        
    def getName(self):
        return self.name
        
    def getStackPos(self):
        return self.stackPos
        
    def trigger(self, basic = True):#this is the method executed by the stack handeller
        if(basic == True):
            print("this is the base Task class and as such has no action")
            return 0
        else:
            return 1

#=============================INDIVIDUAL TASK SUBOBJECTS==========================

class ArmTask(Task):#try and fly an unarmed drone. see how far you get....
    def __init__(self, name, stackpos, connection):
        Task.__init__(self, name, stackpos) 
        self.connection = connection
        #this string tells the code where to look for the microcontroller        
    def trigger(self, connection = None):
        if(connection == None):
            connection = self.connection
        print('Arm Task triggered. Attempting connection through: %s' % connection)
        global vehicle #the object that contains ALL the vehicle info
        vehicle = connect(connection, wait_ready=True) #connect to the MC
        if(vehicle != None):
            print("Connected to vehicle! %s" % vehicle.mode)
        else:
            print "Vehicle not found. Aborting Task..." 
            return 1
        time.sleep(1)
        while not vehicle.is_armable: #this is important, wait until the drone is ready
            print " Waiting for vehicle to initialise..."
            time.sleep(1)
        
        print "Arming motors"
        # Copter should arm in GUIDED mode
        vehicle.mode = VehicleMode("GUIDED") #the drone must be in this mode to understand controls
        vehicle.armed = True    
        time.sleep(1)
        while not vehicle.armed:      
            print " Waiting for arming..."
            time.sleep(1) #better not just ASSUME the vehicle has armed okay
            
        print "Vehicle Armed successfully, Task Complete!"
        time.sleep(1)
        return 0        #see? successful tasks return 0

class TakeoffTask(Task): #generally a good idea to take off before trying anything else...
    def __init__(self, name, stackpos, alt, altErr = 5):
        Task.__init__(self, name, stackpos)
        self.alt = alt #target altitude to rise to
        self.altErr = altErr #well, within this margin of error anyway
        
    def trigger(self, alt = None, altErr = None):
        if(alt == None):
            alt = float(self.alt)
        if(altErr == None):
            altErr = float(self.altErr) #this just makes sure the method can find it's vars
        global vehicle
        print "Takeoff Task Triggered. StackPos: %s Target Alt: %s Error: %s " % (self.stackpos, self.alt, self.altErr)
        time.sleep(3)
        print "Takeoff!"
        vehicle.simple_takeoff(float(alt)) #the magic command itself
        time.sleep(1)
        while ((alt - vehicle.location.global_relative_frame.alt) > altErr):
            print "Climbing... MAV Alt ( %s ) is below target Alt( %s)" % (vehicle.location.global_relative_frame.alt, alt)
            if vehicle.mode.name != "GUIDED": #if the pilot changes the mode the program will exit.
                print "User has changed flight modes - aborting Takeoff"
                return 1
            time.sleep(1) #this loop waits until the vehicle height is higher than the target
        time.sleep(1) #within the margin of error anyway
        print "Now at Target Alt, Task Complete!"
        time.sleep(1)
        return 0        #yay good going return 0 for the win

class WaypointTask(Task):   #GO SOMEWHERE
    def __init__(self, name, stackpos, lat, lon, alt, posErr = 5):
        Task.__init__(self, name, stackpos)
        self.lat = lat
        self.lon = lon
        self.alt = alt
        self.posErr = posErr #same deal but for position
        
    def trigger(self, lat = None,lon = None, alt = None, posErr = None):
        if(lat == None):
            lat = float(self.lat)
        if(lon == None):
            lon = float(self.lon)
        if(posErr == None):
            posErr = float(self.posErr)
        if(alt == None):
            alt = float(self.alt) #get them vars
        global vehicle
        print "Waypoint Task Triggered. StackPos: %s Target Pos: %s, %s Target Alt: %s Error: %s" % (self.stackpos, lat, lon, alt, posErr )
        dest = LocationGlobalRelative(lat, lon, alt) #the dronekit command wants it's data like this
        vehicle.simple_goto(dest) #dronekit command
        time.sleep(1)
        while((math.fabs(lat - vehicle.location.global_relative_frame.lat) > float(metersToDegLat(posErr))) or (math.fabs(lon - vehicle.location.global_relative_frame.lon) > float(metersToDegLon(posErr,vehicle.location.global_relative_frame.lat)))):
            if vehicle.mode.name != "GUIDED": #same deal. pilot quit loop
                print "User has changed flight modes - aborting Waypoint Nav"
                return 1
            #print "Moving... Current position: %s , %s Alt: %s" % (vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon, vehicle.location.global_relative_frame.alt)
            print "Moving... Distance to Target(Lat,Lon): %s , %s" % (math.fabs(lat - vehicle.location.global_relative_frame.lat) , math.fabs(lon - vehicle.location.global_relative_frame.lon))
            #print "posErr: %s" % posErr
            #print (metersToDegLat(5))          
            #print (metersToDegLon(5,vehicle.location.global_relative_frame.lat))
            time.sleep(1)
        #this loop waits until the drone is close enough to where it shoud be
        print "Now at Target Pos, Task Complete!" #yeah you are!
        time.sleep(1)
        return 0
        
class DropTask(Task): #drop that medicine! any injuries caused by falling medicine is your problem
    def __init__(self, name, stackpos, payload):
        Task.__init__(self, name, stackpos)
        self.payload = payload #the first or the second bag?
        
    def trigger(self, payload = None):
        if(payload == None):
            payload = self.payload
        print "Payload Drop Task Activated. Dropping payload No %s" % (payload )
        if(payload == 1):
            msg = vehicle.message_factory.command_long_encode(0,0,183,0,10,1500,0,0,0,0,0)
            vehicle.send_mavlink(msg)
        else:
            msg = vehicle.message_factory.command_long_encode(0,0,183,0,10,1010,0,0,0,0,0)
            vehicle.send_mavlink(msg)
        time.sleep(3)
        return 0
        
class LandTask(Task): #your mission should probably end with this, but you do you.
    def __init__(self, name, stackpos):
        Task.__init__(self, name, stackpos)
        
    def trigger(self):
        global vehicle
        print( "Land Task Activated. Switching to Land mode")
        vehicle.mode = VehicleMode("LAND") #super simple. 
        return 0

class DecisionTask(Task): #ahh...yes... so this allows your code to react to stuff
    def __init__(self, name, stackpos, condition, changecond = None, changestackpos = None ):
        Task.__init__(self, name, stackpos)
        self.condition = condition #this shoud become true when you want to continue
        self.changecond = changecond #or if you want to change where the program is in the stack...
        self.changestackpos = changestackpos #use these two to do it
        
    def trigger(self, condition = None, changecond = None, changestackpos = None):
        if(condition == None):
            condition = self.condition
        if(changecond == None):
            changecond = self.changecond
        if(changestackpos == None):
            changestackpos = self.changestackpos
        print "Decision Task Started"
        while True:
            if (condition == "True"):
                print "Stop condition reached. Progressing..."
                break #did we get the signal to keep going? swell!
            
            if (changecond == "True"): #did we get the signal to change position in the stack?
                global StackPosition
                StackPosition = int(changestackpos)-1 #change the stack indicator to this
                print "Stack Position is now: %s Progressing..." % StackPosition
                break
            
            print "Waiting. Condition Now: %s" % condition
            time.sleep(1) #nothing yet? okay let's wait a little bit
        print"Condition Achieved! Task Complete!"
        time.sleep(1)
        return 0
        
#====================================ACTUAL STACK PROCESSING========================

#Text file Parser
def ParseMission(filestring): #just trust me on this one don't touch it
    print "starting parse"
    missionfile = open(filestring,"r")
    lines = missionfile.readlines()
    #print(filestring)
    objects = []
    num_lines = len(lines)
    n = 3
    while(n < num_lines):
        #print n
        line = lines[n]
        print(line)
        args = line.split()
        tempclass = globals()[args[1]]
        tempargs = [args[0], (n-3) ]
        i = 3
        while(i <= len(args)):
            tempargs.append(args[i-1])
            i+=1

        #print(tempargs)
        objects.append(tempclass(*tempargs))
        n+=1
    return objects
    
#stack handler
    

def FlightStack():      #this is the actual task stack
    print'getting stack'
    global Stack #might want to refer to it elsewhere...
    Stack = ParseMission('TestMission2.txt') #get a LIST of OBJECTS in order.
    global StackPosition
    StackPosition = 0 #THIS TELLS THE PROGRAM WHERE IT IS IN THE MISSION
    while True: #start mission loop
        if(StackPosition == len(Stack)): #are we at the end of the mission?
            print"Out of Tasks. Mission Complete!"
            break #yay end
        print("Triggering Next Task: %s" % StackPosition)
        State = Stack[StackPosition].trigger() #run the trigger method of the current task
        if(State != 0): #RETURN YOUR ZEROES GUYS IT'S NOT HARD
            print "Task Failed. Aborting..."
            vehicle.mode = VehicleMode("ALT_HOLD")
            vehicle.close()
            break
        else:
            StackPosition+=1 #NEXT TASK!
    print"FlightStack is done"
    vehicle.close() #this is super important. you must close the vehicle after you're finished
    
    

#=====================MULTITHREADING AND INITIALISATION===============================    
    
class MAVThread (threading.Thread): #make the stack it's own thread
    def __init__(self,name,threadtype):
        threading.Thread.__init__(self)
        self.name = name
        self.threadtype = threadtype #camera or flight code?
        
    def run(self):
        time.sleep(1)
        print "%s Thread Starting" % self.name
        time.sleep(1)
        if(self.threadtype == "F"):
            FlightStack()
        elif(self.threadtype == "C"):
            CameraLoop()
        else:
            print "unrecognised thread type"
        print"%s Thread Stopping" % self.name

FlightThread = MAVThread("Flight","F")
CameraThread = MAVThread("Camera","C")

FlightThread.start()
#CameraThread.start()
time.sleep(0.5)
print"Main Thread is Dead. Long live StackThreads!"
#FlightTasks
#CameraTasks

