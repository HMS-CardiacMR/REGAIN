######
# Author : Siyeop Yoon
# Last Update : May 26, 2022
# https://cardiacmr.hms.harvard.edu/
######

import pydicom
import pydicom._storage_sopclass_uids
import numpy as np
import glob

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp,QMenuBar, QButtonGroup, QRadioButton, QPushButton
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout,QHBoxLayout,QWidget,QSlider,QLabel
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5 import QtCore

from PyQt5.QtCore import Qt
from PIL import Image
import qimage2ndarray
import matplotlib
matplotlib.use('Qt5Agg')
import cv2

from Config import *
import torch
from torch import Tensor






class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title= "LineDrawer"
        top=400
        left=400
        width=1536
        height=1024


        self.setWindowTitle(self.title)
        self.setGeometry(top,left,width,height)


        self.ButtonLayout = QHBoxLayout()
        self.ButtonLayout.setAlignment(Qt.AlignCenter)
        self.SlideLayout = QHBoxLayout()
        self.SlideLayout.setAlignment(Qt.AlignCenter)

        self.RadioLayout = QHBoxLayout()
        self.RadioLayout.setAlignment(Qt.AlignCenter)


        self.RadioPhases = QVBoxLayout()
        self.RadioPhases.setAlignment(Qt.AlignCenter)


        self.RadioRecons= QVBoxLayout()
        self.RadioRecons.setAlignment(Qt.AlignCenter)


        self.RadioLayout.addLayout(self.RadioPhases)
        self.RadioLayout.addLayout(self.RadioRecons)

        self.ImagesLayout = QHBoxLayout()
        self.ImagesLayout.setAlignment(Qt.AlignCenter)

        self.PageLayout = QVBoxLayout()

        self.PageLayout.addLayout(self.ButtonLayout)
        self.PageLayout.addLayout(self.RadioLayout)

        self.PageLayout.addLayout(self.SlideLayout)
        self.PageLayout.addLayout(self.ImagesLayout)

        self.widget = QWidget()
        self.widget.setLayout(self.PageLayout)



        self.setCentralWidget(self.widget)

        self.ViewMode = 0
        self.SliceMode = 0
        self.PhaseMode = 0
        self.ReconMode = 0
        self.drawingMode = 0


        self.button = QPushButton ("Load Dicoms",self)
        self.button2 = QPushButton("Reconstruct", self)
        self.button3 = QPushButton("Save as Dicom", self)

        self.radioLR = QRadioButton("Phase-encoding : Left-Right ", self)
        self.radioUD = QRadioButton("Phase-encoding : Up-down ", self)

        self.radioReconCurrent= QRadioButton("Reconstruct Current Slice ", self)
        self.radioReconAll = QRadioButton("Reconstruct All Slices", self)



        self.sliceSlider = QSlider(self)
        self.phaseSlider = QSlider(self)
        self.labelslice = QLabel("slice: "+'0', self)
        self.labelslice.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.labelslice.setMinimumWidth(80)

        self.labelphase = QLabel("phase: "+'0', self)
        self.labelphase.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.labelphase.setMinimumWidth(80)


        self.exitAction=QAction('Exit', self)
        self.menubar = QMenuBar()
        self.filemenu = self.menubar.addMenu('&File')

        self.dispay_dim = (800, 800)

        self.maxCardiacPhase=1
        self.maxSlices = 1

        self.image_view1 = QLabel(self)
        self.image_view2 = QLabel(self)

        self.initUI()

    def initUI(self):

        self.ButtonLayout.addWidget(self.button)
        self.ButtonLayout.addWidget(self.button2)
        self.ButtonLayout.addWidget(self.button3)

        self.button.clicked.connect(self.getFolder)
        self.button2.clicked.connect(self.Reconstruction)
        self.button3.clicked.connect(self.saveResults)
        self.btngroupPhases = QButtonGroup()
        self.btngroupPhases.addButton(self.radioLR)
        self.btngroupPhases.addButton(self.radioUD)

        self.RadioPhases.addWidget(self.radioLR)
        self.RadioPhases.addWidget(self.radioUD)
        self.radioLR.toggled.connect(self.updatePhaseMode)
        self.radioUD.toggled.connect(self.updatePhaseMode)

        self.btngroupRecon = QButtonGroup()
        self.btngroupRecon.addButton(self.radioReconCurrent)
        self.btngroupRecon.addButton(self.radioReconAll)

        self.RadioRecons.addWidget(self.radioReconCurrent)
        self.RadioRecons.addWidget(self.radioReconAll)
        self.radioReconCurrent.toggled.connect(self.updateReconMode)
        self.radioReconAll.toggled.connect(self.updateReconMode)

        self.radioLR.toggled.connect(self.updatePhaseMode)
        self.radioUD.toggled.connect(self.updatePhaseMode)

        self.sliceSlider.setOrientation(QtCore.Qt.Horizontal)
        self.sliceSlider.setTickInterval(1)
        self.sliceSlider.setTickPosition(QSlider.TicksBelow)
        self.phaseSlider.setOrientation(QtCore.Qt.Horizontal)
        self.phaseSlider.setTickInterval(1)
        self.phaseSlider.setTickPosition(QSlider.TicksBelow)

        self.phaseSlider.valueChanged.connect(self.updateCurrentFigure)
        self.sliceSlider.valueChanged.connect(self.updateCurrentFigure)
        self.SlideLayout.addWidget(self.sliceSlider)
        self.SlideLayout.addWidget(self.labelslice)
        self.SlideLayout.addWidget(self.phaseSlider)
        self.SlideLayout.addWidget(self.labelphase)

        self.ImagesLayout.addWidget(self.image_view1)
        self.ImagesLayout.addWidget(self.image_view2)

        self.sliceSlider.valueChanged.connect(self.updateLabels)
        self.phaseSlider.valueChanged.connect(self.updateLabels)

        self.exitAction.triggered.connect(qApp.quit)

        self.setMenuBar(self.menubar)
        self.menubar.setNativeMenuBar(True)

        self.filemenu.addAction(self.exitAction)

        img1 = Image.new('RGB', self.dispay_dim, color=(0, 0, 0))
        img1 = np.array(img1)
        img1 = qimage2ndarray.array2qimage(img1)

        self.image_view1.setBackgroundRole(QPalette.Dark)
        self.image_view1.setPixmap(QPixmap.fromImage(img1))

        self.image_view2.setBackgroundRole(QPalette.Dark)
        self.image_view2.setPixmap(QPixmap.fromImage(img1))





        self.ImageResize=4

    lineId=0

    def updatePhaseMode(self):
        phasestr = self.btngroupPhases.checkedButton().text().lower()
        if 'left' in phasestr:
            self.PhaseMode = 0
        elif 'up' in phasestr:
            self.PhaseMode = 1

    def updateReconMode(self):
        reconstr = self.btngroupRecon.checkedButton().text().lower()
        if 'current' in reconstr:
            self.ReconMode = 0
        elif 'all' in reconstr:
            self.ReconMode = 1



    def updateLabels(self):
        self.labelslice.setText("slice: "+str(self.sliceSlider.value()))
        self.labelphase.setText("phase: "+str(self.phaseSlider.value()))




    def getFolder(self):
        self.InFolder = str(QFileDialog.getExistingDirectory(self, "In Patient Directory"))
        self.caseID = os.path.basename(self.InFolder)
        if len(self.InFolder)>0:

            self.maxCardiacPhase=1
            self.maxSlices=1
            self.ndImage = self.ReadFiles(self.InFolder)
            self.ndImage2 =np.array(self.ndImage,copy=True)


            self.updateSliderLimit()
            self.updateDisplaceDim()
            self.updateCurrentFigure()

    def updateSliderLimit(self):
        self.phaseSlider.setMaximum(self.maxCardiacPhase-1)
        self.sliceSlider.setMaximum(self.maxSlices-1)

    def updateDisplaceDim(self):
        self.dispay_dim=(self.ndImage.shape[-2],self.ndImage.shape[-1])

    def ReadFiles(self,InFolder):

        print ("loading "+ InFolder)
        LRs = glob.glob(InFolder + "/*.dcm")
        self.Spatiotemporal_LR = []


        for i in range(len(LRs)):
            pathLR = LRs[i]

            dicom_LR = pydicom.dcmread(pathLR)
            self.Spatiotemporal_LR.append(dicom_LR)

            if self.maxCardiacPhase<dicom_LR.CardiacNumberOfImages:
                self.maxCardiacPhase=dicom_LR.CardiacNumberOfImages

        self.maxSlices=len(LRs)//self.maxCardiacPhase

        self.Spatiotemporal_LR = sorted(self.Spatiotemporal_LR,key=lambda s: s.TriggerTime)
        self.Spatiotemporal_LR = sorted(self.Spatiotemporal_LR,key=lambda s: s.SliceLocation)

        img_LR = self.Spatiotemporal_LR[0].pixel_array.astype(np.float32)

        self.Spatiotemporal_4D=np.zeros((self.maxSlices,self.maxCardiacPhase,img_LR.shape[0],img_LR.shape[1]),dtype=np.float32)

        for islc in range(len(LRs)// self.maxCardiacPhase):

            for iphase in range( self.maxCardiacPhase):
                idx=iphase+islc*self.maxCardiacPhase
                img_LR = self.Spatiotemporal_LR[idx].pixel_array.astype(np.float32)
                slice, _, _=self.PercentileRescaler(img_LR)
                self.Spatiotemporal_4D[islc,iphase,:,:]=slice

        self.Spatiotemporal_SR=self.Spatiotemporal_LR.copy()

        print (np.min(self.Spatiotemporal_4D))
        print(np.max(self.Spatiotemporal_4D))
        print("complete load " + InFolder)
        return self.Spatiotemporal_4D

    def saveResults(self):

        outfolderName="./Output/"+ self.caseID
        if not os.path.exists(outfolderName):
            os.makedirs(outfolderName)

            for islc in range(self.maxSlices):
                for iphase in range(self.maxCardiacPhase):
                    idx = iphase + islc * self.maxCardiacPhase
                    SRDicom=self.Spatiotemporal_SR[idx]
                    img_SR = self.Spatiotemporal_LR[idx].pixel_array.astype(np.float32)
                    _,minval,maxval = self.PercentileRescaler(img_SR)

                    restoredIntensity = self.RestoreRescaler(self.ndImage2[islc,iphase, :, :],minval,maxval)
                    img_16bit=restoredIntensity.astype(np.int16)

                    SRDicom.PixelData = img_16bit.tobytes()
                    SRDicom.save_as(os.path.join(outfolderName, 'recon_Slice' + str(islc)+ "_Phase"+str(iphase)+".dcm"))


    def updateCurrentFigure(self):

        img1 = 255 * np.array(self.ndImage[self.sliceSlider.value(), self.phaseSlider.value(), :, :])
        img1 = cv2.resize(img1, dsize=(img1.shape[1] * self.ImageResize, img1.shape[0] * self.ImageResize), interpolation=cv2.INTER_CUBIC)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)

        img1 = qimage2ndarray.array2qimage(img1)
        self.image_view1.setBackgroundRole(QPalette.Dark)
        self.image_view1.setPixmap(QPixmap.fromImage(img1))

        img2= 255 * np.array(self.ndImage2[self.sliceSlider.value(), self.phaseSlider.value(), :, :])
        img2 = cv2.resize(img2, dsize=(img2.shape[1] * self.ImageResize, img2.shape[0] * self.ImageResize),
                          interpolation=cv2.INTER_CUBIC)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        img2 = qimage2ndarray.array2qimage(img2)

        self.image_view2.setBackgroundRole(QPalette.Dark)
        self.image_view2.setPixmap(QPixmap.fromImage(img2))


    def image2tensor(self, image) -> Tensor:
        tensor = torch.from_numpy(np.array(image, np.float32, copy=False))
        return tensor


    def Reconstruction (self):

        if self.ReconMode==0:

            Image = self.ndImage[self.sliceSlider.value(), self.phaseSlider.value(), :, :]
            IsLeftRight = (self.PhaseMode == 0)

            if IsLeftRight :
                Image = Image.transpose()

            lr_image,_,_ =  self.PercentileRescaler(Image)
            lr_tensor = self.image2tensor(lr_image).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                sr_tensor1 = generator(lr_tensor)
                sr_tensor1 = sr_tensor1.squeeze()
                img = torch.from_numpy(np.array(sr_tensor1.to('cpu'), np.float32, copy=False))
                img = np.array(img)
                img = np.clip(img, 0, 1)

                if IsLeftRight:
                    img = img.transpose()

                self.ndImage2[self.sliceSlider.value(), self.phaseSlider.value(), :, :]=img

        elif self.ReconMode==1:

            for phases in range (self.maxCardiacPhase):
                for slice in range(self.maxSlices):
                    Image = self.ndImage[slice, phases, :, :]
                    IsLeftRight = (self.PhaseMode == 0)

                    if IsLeftRight:
                        Image = Image.transpose()

                    lr_image,_,_ = self.PercentileRescaler(Image)
                    lr_tensor = self.image2tensor(lr_image).unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        sr_tensor1 = generator(lr_tensor)
                        sr_tensor1 = sr_tensor1.squeeze()
                        img = torch.from_numpy(np.array(sr_tensor1.to('cpu'), np.float32, copy=False))
                        img = np.array(img)
                        img = np.clip(img, 0, 1)
                        if IsLeftRight:
                            img = img.transpose()

                        self.ndImage2[slice, phases, :, :] = img

        self.updateCurrentFigure()

    def PercentileRescaler(self,Arr):
        minval = np.percentile(Arr, 0, axis=None, out=None, overwrite_input=False, interpolation='linear',
                               keepdims=False)
        maxval = np.percentile(Arr, 100, axis=None, out=None, overwrite_input=False, interpolation='linear',
                               keepdims=False)

        Arr = (Arr - minval) / (maxval - minval)
        Arr = np.clip(Arr, 0.0, 1.0)
        return Arr, minval,maxval

    def RestoreRescaler(self,Arr, minval, maxval):
        arr = Arr * (maxval - minval) + minval
        arr = np.clip(arr, 0.0, maxval)
        return arr







if __name__ == '__main__':

    app = QApplication(sys.argv)  # QApplication eats argv in constructor
    window = MainWindow()
    window.setWindowTitle('Viewer Project')
    window.show()
    sys.exit(app.exec_())








