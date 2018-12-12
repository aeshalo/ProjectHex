This section Collects some of the work I did for Project Hex during my time in the project

Please Note that this is not the actual Project Hex Repository. After leaving, access was transferred to the new control team.

In this repository the main files of note are:

NavDraft1
This file contains the complete navigation and computer vision code used to control the drone. 
It's multithreaded for saftey, to ensure mission-critical processes are never made to wait for non-critical work (camera),
and is object oriented for easy extendability, modularity and mission flexibility.

This code uses a Task Sheduling method for missions
Each task is instantiated as an Object, with a method that differs depending on the task.
In this way Task code can be reused Multiple times
A pointer is used to keep track of the current task. Every time a task finishes it's method, the pointer increases and the next task's method is called
several Task methods loop until certan conditions are met, such as drone position or a simple wait timer.




It's Primary Libraries are Dronekit and OpenCV
Dronekit handles all communication between the python script and the ardupilot microcontroller.
OpenCV handles all the computer Vision Tasks.

Several pictures and videos are included giving additional information or showing the drone in action, running NavDraft1

CVTest is a prior version of the computer vision code, this was eventually folded into NAvDraft1
