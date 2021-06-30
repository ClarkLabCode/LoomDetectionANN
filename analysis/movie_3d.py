"""
In this file, the 3d reconstruction of moving objects and the receptive fields of model units are simulated.
Adapted from https://github.com/mcfletch/pyopengl.
"""

import OpenGL 
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time, sys
import os
import numpy as np
import imageio
from PIL import Image

import optical_signal as opsg

figure_path = '/Volumes/Baohua/research/loom_detection/results/final_figures_for_paper_exp/'
if not os.path.exists(figure_path + 'movies/movie_frames_3d/'):
    os.makedirs(figure_path + 'movies/movie_frames_3d/')

OpenGL.ERROR_ON_COPY = True 


def drawCone(position=(1, 0, -5+11), orientation=(0, 0, 0, 0), radius=1, height=5, slices=50, stacks=10):
    glPushMatrix()
    try:
        glTranslatef(*position)
        glRotatef(*orientation)
        glutWireCone(r, h, slices, stacks)
    finally:
        glPopMatrix()


def drawCones(position=(0, 0, -5), orientation=(0, 0, 0, 0), radius=1, height=5, slices=50, stacks=10):
    glPushMatrix()
    try:
        glRotatef(*orientation)
        glTranslatef(*position)
        glutSolidCone(r, h, slices, stacks )
    finally:
        glPopMatrix()
         
            
def drawSphere(sphere, R=0.1, x=0, y=0, z=0, orientation=(0, 0, 0, 0)):
    glPushMatrix()
    try:
        glTranslatef(x, y, z)
        glColor4f(0.5, 0., 0., 1) #Put color
        gluSphere(sphere, R, 32, 16) #Draw sphere
    finally:
        glPopMatrix()


def coneMaterial( ):
    """Setup material for cone"""
#     glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, .1))
#     glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.5, 0.5, 0.5, .1))
#     glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(1.0, 0.0, .0, 1.0))
#     glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(1.0))
def light():
    """Setup light 0 and enable lighting"""
    glLightfv(GL_LIGHT0, GL_AMBIENT, GLfloat_4(0.5, 0.5, 0.5, .1))
#     glLightfv(GL_LIGHT0, GL_DIFFUSE, GLfloat_4(1.0, 1.0, 1.0, 1.0))
#     glLightfv(GL_LIGHT0, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
#     glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(1.0, 1.0, 1.0, 1.0));   
#     glLightModelfv(GL_LIGHT_MODEL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
def depth():
    """Setup depth testing"""
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)

def display(swap=1, clear=1):
    """Callback function for displaying the scene

    This defines a unit-square environment in which to draw, 
    i.e. width is one drawing unit, as is height
    """
    t = 0
    combined_movies_list = []
    while t < T: 
        if clear:
            glClearColor(0, 0, 0, 0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # establish the projection matrix (perspective)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        x, y, width, height = glGetDoublev(GL_VIEWPORT)
        gluPerspective(
            45, # field of view in degrees
            width/float(height or 1), # aspect ratio
            .25, # near clipping plane
            200, # far clipping plane
        )
        glRotatef(0, 1, 1, 0)
        glTranslatef(0, -2, -20)

        # and then the model view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            0, 1, 5, # eyepoint
            0, 0, 0, # center-of-view
            0, 1, 0, # up-vector
        )
        light()
        depth()
        coneMaterial()

        rotation(starttime)
        
        if M == 1:
            drawCone(orientation=(1, 1, 1, 0))
        else:
            for coords in lplc2_units_coords:
                vec_r = np.cross([0, 0, -1], coords)
                angle = opsg.get_angle_two_vectors([0, 0, -1], coords)*180/np.pi
                drawCones(orientation=(angle, *vec_r))

        sphere = gluNewQuadric()
        R = 1
        P = len(traj[t])
        for p in range(P):
            x, y, z = traj[t][p]
            x = x+1
            z = -z+11
            drawSphere(sphere, Rs[p], x, y, z)
        swap = 0 # setting swap to be 0 causes no showing of the movies, but makes saving correct.
        if swap:
            glutSwapBuffers()
        save_path = figure_path + 'movies/stimuli/movie_frames_3d/'+data_type+'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = save_path+'image3d_{}'.format(t+1)
        saveBuffer(filename, 'png')
        t = t+1
        if t == T:
            sys.exit()
            

def idle( ):
    glutPostRedisplay()

def rotation(starttime, period = 10):
    """Do rotation of the scene at given rate"""
    angle = (((time.time()-starttime)%period)/period)* 360
    glRotate( -90, 0, 0, 1)
    glRotate( rot_angle, rot_x, rot_y, rot_z)
    return angle

def saveBuffer( filename, format1 ):
    """Save current buffer to filename in format"""
    x, y, width, height = glGetDoublev(GL_VIEWPORT)
    width, height = int(width/2), int(height/2)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes( "RGB", (width, height), data )
    image = image.transpose( Image.FLIP_TOP_BOTTOM)
    image.save(filename, format1, dpi=(300, 300))

    
from OpenGL._bytes import as_8_bit
ESC = as_8_bit( '\033' )
def key_pressed(*args):
    # If escape is pressed, kill everything.
    if args[0] == ESC:
        sys.exit()

def main():    
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow('Rotating Cone')
    glutDisplayFunc(display)
    glutKeyboardFunc(key_pressed)
    glutIdleFunc(display)
    # note need to do this to properly render faceted geometry
    glutMainLoop()

if __name__ == "__main__":
    
    starttime = time.time()
    M = int(sys.argv[1])
    data_type_number = int(sys.argv[2])
    rot_angle = int(sys.argv[3])
    rot_x = int(sys.argv[4])
    rot_y = int(sys.argv[5])
    rot_z = int(sys.argv[6])
    
    if M == 1:
        lplc2_units_coords = np.array([[0, 0, -1]])
    else:
        _, lplc2_units_coords = opsg.get_lplc2_units(M)

    h = 5
    r = h*np.tan(30*np.pi/180)

    data_types = ['hit', 'miss', 'retreat', 'rotation']
    data_type = data_types[data_type_number]
    data_path = figure_path + 'movies/stimuli/movie_frames/'+data_type+'/'
    traj = np.load(data_path+'traj.npy')
    
    if data_type_number == 3:
        Rs = np.load(data_path+'Rs.npy') / 3
        traj = traj / 3
    else:
        Rs = [1]
    T = len(traj)
    print(f'Total length of the movie is {T}.')
    angle_0 = opsg.get_angle_two_vectors([0, 0, 1], traj[1][0]-traj[0][0])*180/np.pi
    if angle_0<=180:
        main()

