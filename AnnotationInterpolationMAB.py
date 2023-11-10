#!/usr/bin/env python
# coding: utf-8

#%%
# Import libraries
get_ipython().run_line_magic('matplotlib', 'ipympl')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pydicom import dcmread
import os, glob


# Read DICOM annotations

ANNOTATION_DIR = "/home/ERASMUSMC/099035/Documents/AnnotationFiles"

#%%
def list_annotations(annotation_dir,filename_expression = '*.dcm'):
    filenames = glob.glob(os.path.join(annotation_dir,filename_expression))
    filenames = sorted(filenames)
    print(f'There are {len(filenames)} annotations in the directory "{annotation_dir}"')
    return filenames
#%%
# Read filenames and find total number for files
filenames = list_annotations(ANNOTATION_DIR)
filenames_size = len(filenames)


#%%
# Read the annotations metadata from a single file
annotations_meta = dcmread(filenames[0])
annotations_meta['ApplicationSetupSequence'].value
#OR using the tag value
#annotations_meta[0x300A,0x0230].value

# Printouts that helped me understand the data
#annotations_meta.ApplicationSetupSequence[0].dir()
#channel_seq = annotations_meta.ApplicationSetupSequence[0].ChannelSequence
#channel_seq
#sequence = annotations_meta[0x300A,0x0230].value
#sequence[0].ChannelSequence
#annotations_meta.ApplicationSetupSequence[0].ChannelSequence[0].BrachyControlPointSequence[41].ControlPoint3DPosition


#%%
# Create a class for the Dwell Data
class DwellData:
    channel = np.array([np.nan])
    position = np.array([np.nan])
    coordinates = np.array([0])

    def __init__(self,filename):
        annotations_metadata = dcmread(filename)
        # ApplicationSetupSequence is of Class type 'Sequence' from pydicom docs
        channel_seq = annotations_metadata.ApplicationSetupSequence[0].ChannelSequence
        k=0
        for c in range(0,len(channel_seq)):
            control_point_seq = channel_seq[c].BrachyControlPointSequence
            relative_position = float('NaN')
            # Initialize arrays if its the first pass
            # Checks if any of the values evaluates to True (non zero)
            is_nan = np.isnan(self.channel)
            if (np.all(is_nan) == True):
                # Initialize numpy arrays with zeros (1D,1D,3D)
                self.channel = np.full((len(channel_seq) * len(control_point_seq)), np.nan)
                self.position = np.full((len(channel_seq) * len(control_point_seq)), np.nan)
                self.coordinates = np.zeros([len(channel_seq) * len(control_point_seq),3])
            # Loop in the BrachyControlPointSequence from index len(seq)-1 to 0
            for i in range(len(control_point_seq)-1,-1,-1):
                if (control_point_seq[i].ControlPointRelativePosition!=relative_position):
                    relative_position = control_point_seq[i].ControlPointRelativePosition
                    self.channel[k] = c + 1
                    self.position[k] = relative_position
                    self.coordinates[k,:] = control_point_seq[i].ControlPoint3DPosition
                    k+=1
        self.channel = self.channel[np.isnan(self.channel) == False].astype('int32')
        self.position = self.position[np.isnan(self.position) == False].astype('int32')
        self.coordinates = self.coordinates[self.coordinates != 0].reshape((-1,3))
            


#%%
# Create an object of class DwellData
PointData = DwellData(filenames[0])
#PointData
#PointData.coordinates[0]


#%%
# Plot the loaded annotations

fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.scatter3D(PointData.coordinates[:,0],PointData.coordinates[:,1],PointData.coordinates[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Annotations")
plt.show()

#%%

# Curve Fitting Section
# Import libraries
from scipy.optimize import curve_fit

#%%
# Define mathematical function
# Polynomial function of degree 3
# Will use this function for every channel i.e every catheters points
def func(xy,a,b,c,d):
    x, y = xy
    #return a + b*x + c*y + d*x**2 + e*y**2 + f*x*y
    return a + b*x + c*y + d*x*y

# Calculate curve fit parameters
def calculate_curve_fit(func,x,y,z):
    popt, pcov = curve_fit(func,(x,y),z,p0=[1 , 0.5 , 0.5,0.5])#method = 'trf'
    return popt,pcov

# Correlate coordinates with channel
# V2 is faster
def match_coordinate_channel_V2(point_data):
    channel_length = len(np.unique(point_data.channel))
    number_of_coordinates_per_channel = int(point_data.coordinates.shape[0]/channel_length)
    channel_coordinates = np.reshape(point_data.coordinates,(channel_length,number_of_coordinates_per_channel,3))
    return channel_coordinates

def match_coordinate_channel_V1(point_data):
    channel_length = len(np.unique(point_data.channel))
    channel_coordinates = np.zeros((channel_length,int(point_data.coordinates.shape[0]/channel_length),3))
    #print(channel_coordinates.shape)
    for i in range(0,channel_length):
        temp_channel = np.where(point_data.channel == i+1,True,False)
        #print(temp_channel)
        channel_coordinates[i,:,:] = point_data.coordinates[temp_channel]
    return channel_coordinates


#%%
# Test the speed of the two functions
get_ipython().run_line_magic('timeit', 'channel_coordinates = match_coordinate_channel_V2(PointData)')
get_ipython().run_line_magic('timeit', 'channel_coordinates = match_coordinate_channel_V1(PointData)')

#%%
# Find max and min values of a channel
def find_min_max_single(single_channel_coordinates):
    X_max = np.max(single_channel_coordinates[:,0])
    X_min = np.min(single_channel_coordinates[:,0])
    Y_max = np.max(single_channel_coordinates[:,1])
    Y_min = np.min(single_channel_coordinates[:,1])
    return X_max,X_min,Y_max,Y_min

# Find max and min values of all channels
def find_min_max_all(all_channel_coordinates):
    channel_length = all_channel_coordinates.shape[0]
    X_max = np.zeros((channel_length))
    X_min = np.zeros((channel_length))
    Y_max = np.zeros((channel_length))
    Y_min = np.zeros((channel_length))
    for i in range(0,channel_length):
        X_max[i],X_min[i],Y_max[i],Y_min[i] = find_min_max_single(all_channel_coordinates[i,:,:])
    return X_max,X_min,Y_max,Y_min


#%%
# Match coordinates to their channels
channel_coordinates = match_coordinate_channel_V2(PointData)
# Find max and min values from X and Y coordinates
channel_coordinates_max_min = np.array(find_min_max_all(channel_coordinates))
channel_coordinates_max_min = channel_coordinates_max_min.T
# channel_coordinates_max_min columns are X_max,X_min,Y_max,Y_min
Z_fitted = np.zeros((channel_coordinates.shape[0],100))
x_ranges = np.zeros((channel_coordinates.shape[0],100))
y_ranges = np.zeros((channel_coordinates.shape[0],100))
popt = np.zeros((channel_coordinates.shape[0],4))
pcov = np.zeros((channel_coordinates.shape[0],4,4))
for i in range(0,channel_coordinates.shape[0]):
    # Perform Curve fitting
    x = channel_coordinates[i,:,0] #/ np.max(channel_coordinates[i,:,0])
    y = channel_coordinates[i,:,1] #/ np.max(channel_coordinates[i,:,1])
    z = channel_coordinates[i,:,2] #/ np.max(channel_coordinates[i,:,2])
    popt[i,:], pcov[i,:,:] = calculate_curve_fit(func,x,y,z)
    # Define range of curve fit line
    x_range = np.linspace(channel_coordinates_max_min[i,1],channel_coordinates_max_min[i,0],100)
    y_range = np.linspace(channel_coordinates_max_min[i,3],channel_coordinates_max_min[i,2],100)
    x_ranges[i,:] = x_range
    y_ranges[i,:] = y_range
    #x_range, y_range = np.meshgrid(x_range, y_range)
    # Calculate Z from X and Y coordinates
    Z_fitted[i,:] = func((x_range,y_range), *popt[i,:])
    #Z_fitted[i,:] = func((X,Y), *popt)
Z_fitted.shape


#%%
# Analyze optimization parameters to determine why fit isn't good
popt[0,:]
np.linalg.cond(pcov[:,:])
np.diag(pcov[0,:,:])

#%%
# Plot the curve fitted catheters
fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.scatter3D(PointData.coordinates[:,0],PointData.coordinates[:,1],PointData.coordinates[:,2])
# For all channels (catheters)
for i in range(0,Z_fitted.shape[0]):
    ax.scatter3D(x_ranges[i,:],y_ranges[i,:],Z_fitted[i,:], color="red")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Annotations")
plt.show()

#%%
# Try new methods : RBFinterpolator
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import SmoothBivariateSpline
# Match coordinates to their channels
channel_coordinates = match_coordinate_channel_V2(PointData)
# Find max and min values from X and Y coordinates
channel_coordinates_max_min = np.array(find_min_max_all(channel_coordinates))
channel_coordinates_max_min = channel_coordinates_max_min.T
# channel_coordinates_max_min columns are X_max,X_min,Y_max,Y_min
Z_fitted = np.zeros((channel_coordinates.shape[0],100))
for i in range(0,channel_coordinates.shape[0]):
    # Perform Curve fitting
    x = channel_coordinates[i,:,0] #/ np.max(channel_coordinates[i,:,0])
    y = channel_coordinates[i,:,1] #/ np.max(channel_coordinates[i,:,1])
    #xy = np.vstack((x,y))
    #print(xy.shape)
    #print(xy[0,0],xy[1,0])
    #xy = xy.T
    #print(xy.shape)
    #print(xy[0,0],xy[0,1])
    z = channel_coordinates[i,:,2] #/ np.max(channel_coordinates[i,:,2])
    weights = np.zeros((channel_coordinates[i,:,0].shape))
    #spline = RBFInterpolator(xy,z,kernel = 'cubic')
    spline = SmoothBivariateSpline(x,y,z, s=0.0)
    x_range = np.linspace(channel_coordinates_max_min[i,1],channel_coordinates_max_min[i,0],100) #/ np.max(channel_coordinates[i,:,0])
    y_range = np.linspace(channel_coordinates_max_min[i,3],channel_coordinates_max_min[i,2],100) #/ np.max(channel_coordinates[i,:,1])
    x_ranges[i,:] = x_range
    y_ranges[i,:] = y_range
    Z_fitted[i,:] = spline.ev(x_range,y_range)    


#%%
# Plot the curve fitted catheters
fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.scatter3D(PointData.coordinates[:,0],PointData.coordinates[:,1],PointData.coordinates[:,2])
# For all channels (catheters)
for i in range(0,Z_fitted.shape[0]):
    ax.scatter3D(x_ranges[i,:],y_ranges[i,:],Z_fitted[i,:], color="red")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("Annotations")
#ax.view_init(90)
plt.show()


#%%
# Try new methods : splprep,splev

from scipy.interpolate import splprep, splev
x = channel_coordinates[0,:,0] #/ np.max(channel_coordinates[i,:,0])
y = channel_coordinates[0,:,1] #/ np.max(channel_coordinates[i,:,1])
z = channel_coordinates[0,:,2] #/ np.max(channel_coordinates[i,:,2])
tck, u = splprep([x,y,z],s=0)
#spl = splprep([x,y,z],s=0)
#new_points = splev(u, tck)
#print(new_points[0])

x_range = np.linspace(channel_coordinates_max_min[0,1],channel_coordinates_max_min[0,0],100) #/ np.max(channel_coordinates[i,:,0])
y_range = np.linspace(channel_coordinates_max_min[0,3],channel_coordinates_max_min[0,2],100) #/ np.max(channel_coordinates[i,:,1])
new_points = splev(u, tck)

# Plot
fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.scatter3D(PointData.coordinates[0:21,0],PointData.coordinates[0:21,1],PointData.coordinates[0:21,2],'b')
ax.plot(new_points[0], new_points[1],new_points[2], 'r-')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


#%%
# Try new methods : BSpline

from scipy.interpolate import BSpline
t, c, k = tck
c1 = np.asarray(c)
spl = BSpline(t, c1.T, k)
z_new = spl(x_range)
z_new.shape
fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.scatter3D(PointData.coordinates[0:21,0],PointData.coordinates[0:21,1],PointData.coordinates[0:21,2],'b')
ax.plot(z_new[:,0],z_new[:,1],z_new[:,2], 'r.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()


#%%
# Try new methods : bisplrep, bisplev
from scipy.interpolate import bisplrep, bisplev
x = channel_coordinates[0,:,0] #/ np.max(channel_coordinates[i,:,0])
y = channel_coordinates[0,:,1] #/ np.max(channel_coordinates[i,:,1])
z = channel_coordinates[0,:,2] #/ np.max(channel_coordinates[i,:,2])
tck = bisplrep(x,y,z,s=0)

#print(new_points[0])

x_range = np.linspace(channel_coordinates_max_min[0,1],channel_coordinates_max_min[0,0],100) #/ np.max(channel_coordinates[i,:,0])
y_range = np.linspace(channel_coordinates_max_min[0,3],channel_coordinates_max_min[0,2],100) #/ np.max(channel_coordinates[i,:,1])

znew = bisplev(x_range,y_range, tck)

fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.scatter3D(PointData.coordinates[0:21,0],PointData.coordinates[0:21,1],PointData.coordinates[0:21,2],'b')
X,Y = np.meshgrid(x_range,y_range)
ax.plot_surface(X,Y,znew,color='red',alpha=0.5)
#ax.plot(x_range[50], y_range[50], znew[0,50], 'r.')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()