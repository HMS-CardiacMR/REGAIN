
"""
Author: Siyeop Yoon and Manuel Morales
Affiliation : Cardiac MR center at Beth Israel Deaconess Medical Center and Harvard Medical School
"""


"""
Implementation of

Ahmad R, Ding Y, Simonetti OP.
Edge sharpness assessment by parametric modeling: application to magnetic resonance imaging.
Concepts Magn Reson 2015;44(3):138-149. doi: DOI: 10.1002/cmr.a.21339
"""

import numpy as np
import cv2
import scipy.ndimage



def sigmoid(x, s, a0, a1, a2):
    """ Sigmoid function to quantify sharpness.
        s:  Quantifies the growth rate or slope of the sigmoid.
        a0: Determines the center location.
        a1: Determines the vertical range.
        a2: Defines the vertical offset.
    """
    y = a1 / (1 + np.exp(-s * (x - a0))) + a2

    return y

def fit2sigmoid(x, y):
    """ Fit y to sigmoid function.
    """
    from scipy.optimize import curve_fit

    p0 = [1,  0, max(y)-min(y),min(y)] # initial guess

    popt, pcov = curve_fit(sigmoid, x, y, p0, method='trf', maxfev=50000)

    return popt, pcov, sigmoid(x, *popt)


def PercentileRescaler_min_max(Arr):
    minval = np.percentile(Arr, 0, axis=None, out=None, overwrite_input=False, interpolation='linear',
                           keepdims=False)
    maxval = np.percentile(Arr, 100, axis=None, out=None, overwrite_input=False, interpolation='linear',
                           keepdims=False)

    Arr = (Arr - minval) / (maxval - minval)
    Arr = np.clip(Arr, 0.0, 1.0)
    return Arr, minval, maxval

def Sharpness_exp(inputImage, Center, SegPt1, SegPt2):
    inputImage_norm, minval, maxval = PercentileRescaler_min_max(inputImage)

    dist2ctr1 = np.linalg.norm(np.array(SegPt1) - np.array(Center)) # get distance to select the correct start point
    dist2ctr2 = np.linalg.norm(np.array(SegPt2) - np.array(Center)) # get distance to select the correct start point

    #Detect Start and End of segment
    if dist2ctr1 > dist2ctr2:
        startpt = SegPt1
        endpt = SegPt2
    else:
        startpt = SegPt2
        endpt = SegPt1

    # segment length
    dist1 = (np.linalg.norm(np.array(endpt) - np.array(startpt)))

    # number of samples along the segment to be extracted
    num = int(50)
    # Sampling points along segments
    x = np.linspace(startpt[0], endpt[0], num)
    y = np.linspace(startpt[1], endpt[1], num)

    # Extract Image Profile
    IntensityProfile = scipy.ndimage.map_coordinates(
        np.transpose(inputImage_norm),
        np.vstack((x, y)))

    # Distance in image.
    x_fit = np.linspace(-dist1/2, dist1/2, num)
    a, b, fitC = fit2sigmoid(x_fit, IntensityProfile) #Sigmoid fitting

    Sigmoid_Slope=abs(a[0])

    return Sigmoid_Slope

"""
Implementation of

McCann A, Workman A, McGrath C. 
A quick and robust method for measurement of signal-to-noise ratio in MRI. 
Physics in Medicine & Biology 2013;58(11):3775. 

"""
def SNR_by_SIS(Image, Center, Rad,kernelSize):
    # Get Mean signal
    ImageMask = np.zeros((Image.shape[0], Image.shape[1], 1), dtype=np.uint8)
    ImageMask = cv2.circle(ImageMask, Center, Rad, 1, -1)
    Mean, Std = cv2.meanStdDev(Image, mask=ImageMask)

    # Boxcar Filtering
    kernel = np.ones((kernelSize,kernelSize)) / (kernelSize*kernelSize)
    smoothed_image=scipy.ndimage.convolve(Image,kernel)

    # Substracted Image
    sis=Image-smoothed_image

    # Get std of substracted Image
    SISMask = np.zeros((Image.shape[0], Image.shape[1], 1), dtype=np.uint8)
    SISMask = cv2.circle(SISMask, Center, Rad, 1, -1)
    SISMean, SISStd = cv2.meanStdDev(sis, mask=SISMask)

    return Mean / SISStd

if __name__ == "__main__":
    dataroot="./data.npy"
    Image = np.load(dataroot)


    #######
    # SNR #
    #######
    #Manually selected Points in Image
    BloodCenter = (125, 125)
    BloodRad = 9

    # Manually selected Points in Image
    MyoCenter = (100, 100)
    MyoRad = 4

    SNR_Blood = SNR_by_SIS(Image, BloodCenter, BloodRad)
    SNR_Myo   = SNR_by_SIS(Image, MyoCenter, MyoRad)

    #############
    # Sharpness #
    #############
    Slopes = []

    # Manually selected Points in Image
    # 1st segment
    SegPt1 = (102, 102) # start from myocardium
    SegPt2 = (122, 122) # end to blood pool


    Slope1= Sharpness_exp(Image, BloodCenter, SegPt1, SegPt2)
    Slopes.append(Slope1)

    # ...


    # 6th segment
    Slope6 = Sharpness_exp(Image, BloodCenter, SegPt1, SegPt2)
    Slopes.append(Slope6)

    asArray = np.asarray(Slopes)
    image_sharpness = np.mean(asArray, axis=0)


