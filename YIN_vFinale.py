# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:56:04 2021

@author: rapha
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.fftpack import fft, ifft

import time




def nextpow2(i):
    '''
    Find the next power 2 number for FFT
    '''
    n = 1
    while n < i: n *= 2
    return n

def shift_signal_in_frequency_domain(datin, shift):
    '''
    This is function to shift a signal in frequency domain. 
    The idea is in the frequency domain, 
    we just multiply the signal with the phase shift. 
    '''
    Nin = len(datin) 
    # get the next power 2 number for fft
    N = nextpow2(Nin +np.max(np.abs(shift)))
    # do the fft
    fdatin = np.fft.fft(datin, N)
    # get the phase shift for the signal, shift here is D in the above explaination
    ik = np.array([2j*np.pi*k for k in range(0, N)]) / N 
    fshift = np.exp(-ik*shift)
    # multiple the signal with the shift and transform it back to time domain
    datout = np.real(np.fft.ifft(fshift * fdatin))
    # only get the data have the same length as the input signal
    datout = datout[0:Nin]
    return datout    
    
def FAC(x,N,t0,W,Fs):
    """STEP 1 : Autocorrelation fonction (Eq 1)
    
    param x : signal to be analyzed (1D numpy array)
    param N : lenght of the signal (int)
    param t0 : start time of the signal (float)
    param W : lenght of the window (int)
    param Fs : Sampling frequency of the signal (float)
    
    return : the autocorrelation of the windowed signal (from t0 to t0+W)

    """ 
    indbeg = t0                    # index of bginning of the current frame
    indend = np.max([[t0+W],[N]])                     # index of end of the current frame
    xtmp = x[indbeg:indend]         # extraction of corresponding x values
    # estimation of the (not shifted) ACF #
    # ----------------------------------- #
    corr = np.real(ifft(np.abs(fft(xtmp))))
    return corr


def DiffFunction(x,N,t0,W,Fs):
    """STEP 2 : Difference fonction (Eq7)
    
    param x : signal to be analyzed (1D numpy array)
    param N : lenght of the signal (int)
    param t0 : start time of the signal (float)
    param W : lenght of the window (int)
    param Fs : Sampling frequency of the signal (float)
    
    return : the difference function of the windowed signal (from t0 to t0+W)
    note : quite long to execute, is optimized in "DiffFunction_new"
    
    """    
    term3 = np.zeros(N,dtype=float)
    for n in range(N):
        indbeg = np.max([[0],[n-int(W/2)]])
        indend = np.min([[N],[n+int(W/2)]])
        term3[n] = np.sum(x[indbeg:indend]**2)
    
    # the calculation for frame l #
        # ----------------------- #

    indbeg = t0                   # index of bginning of the current frame
    indend = t0+W                # index of end of the current frame
    xtmp = x[indbeg:indend]         # extraction of corresponding x values
    # estimation of the (not shifted) ACF #
    # ----------------------------------- #
    Rxxtmp = np.real(ifft(np.abs(fft(xtmp))**2))
    # dtmp corresponds to d_t(\tau) in step 2 #
    # --------------------------------------- #
    df = np.zeros(int(W/2),dtype=float)
    for n in range(int(W/2)):
        df[n] = Rxxtmp[0]-2*Rxxtmp[n]+term3[t0+int(W/2)]
    # what you now have to do is to estimate the value of n corresponding to the first min
    # and to deduce the local period or equivalently the local frequency
    return df

def DiffFunction_new(x,N,t0,W,Fs):
    """STEP 2 : Difference fonction (Eq7) using filter to increase the speed of the computation
    
    param x : signal to be analyzed (1D numpy array)
    param N : lenght of the signal (int)
    param t0 : start time of the signal (float)
    param W : lenght of the window (int)
    param Fs : Sampling frequency of the signal (float)
    
    return : the difference function of the windowed signal (from t0 to t0+W)

    
    """    
    b = np.ones(W)
    term3 = lfilter(b,[1.],x**2)    
    term3 = shift_signal_in_frequency_domain(term3,-W+1)  
    
    # the calculation for frame l #
        # ----------------------- #
    t0 = int(t0*Fs)
    indbeg = t0                   # index of bginning of the current frame
    indend = t0+W                # index of end of the current frame
    xtmp = x[indbeg:indend]         # extraction of corresponding x values
    # estimation of the (not shifted) ACF #
    # ----------------------------------- #
    Rxxtmp = np.real(ifft(np.abs(fft(xtmp))**2))
    # dtmp corresponds to d_t(\tau) in step 2 #
    # --------------------------------------- #
    df = np.zeros(int(W/2),dtype=float)
    for n in range(int(W/2)):
        df[n] = Rxxtmp[0]-2*Rxxtmp[n]+term3[t0+int(W/2)]
    # what you now have to do is to estimate the value of n corresponding to the first min
    # and to deduce the local period or equivalently the local frequency
    return df-df[0]   #-df[0] is a correction to step the beginning at 0 (probably a bit 'savage')

def CumulDiff(df) :
    """STEP 3 : Cumulative mean normalized difference function (Eq.8)
    
    param df : the difference function to be treated (1D numpy array)
    
    return : The Cumulative mean normalized difference function (1D numpy array)
    
    """
    cmndf = df[1:] * range(1, len(df)) / np.cumsum(df[1:]).astype(float) #scipy method
    return np.insert(cmndf, 0, 1)

def Threshold(cmndf):
    """STEP 4 : Absolute threshold
    
    param cmdf : The Cumulative mean normalized difference function (1D numpy array)
    
    return : The computed argument of the period (int)
    
    Note :  Return the first dip included in the threshold defined by 90% of the difference between the mean and the min of the cmndf
            Might return 0 and cause troubles later
            
    """  
    th = np.mean(cmndf) - (np.mean(cmndf)-np.min(cmndf))*0.90
    estimate_tau = 0
    if np.min(cmndf)<th:
        while cmndf[estimate_tau]>th:
            estimate_tau += 1
        while cmndf[estimate_tau]>cmndf[np.min([[estimate_tau+1],[len(cmndf)-1]])]:
            estimate_tau +=1
        return estimate_tau
    
    else: return np.argmin(cmndf)
 
def Interpol(df,Fs,estimate_tau):
    """STEP 5 : Parabolic Interpolation
    
    param df : The difference function (1D numpy array)
    Fs : The sampling frequency (float)
    estimate_tau : The first estimation of Threshold (int)
    
    return : A better approximation of the local minimum

    note : may fail for minimums on the edges of cmndf
    
    """
    #Fifth method (Theory : 0.77% error)
    #Local minimum for t = nT
    #Take as input the variable, function and the value of the first dip according to Threshold
    tau = [i/Fs for i in range(len(df))]
    indbeg = np.max([[0] ,[estimate_tau-3]])
    indend = np.min([[len(df)],[estimate_tau+4]])
    tau = tau[indbeg:indend]
    cmndf = df[indbeg:indend]
    z = np.polyfit(tau,cmndf,3)
    p = np.poly1d(z)
    crit = p.deriv().r
    r_crit = crit[crit.imag==0].real
    test = p.deriv(2)(r_crit) 


    # compute local minima 
    # excluding range boundaries
    x_min = r_crit[test>0]
    
    if len(x_min) != 0:
        return x_min
    elif x_min[0] < 0:
        return 0
    else: return 0
    

def YIN(x,N,t0,W,Fs,fmin,fmax):
    
    """YIN Method
    
    param x : signal to be analyzed (1D numpy array)
    param N : lenght of the signal (int)
    param t0 : start time of the signal (float)
    param W : lenght of the window (int)
    param Fs : Sampling frequency of the signal (float) 
    para fmin : The minimum frequency of the signal (float)
    para fmax : The maximum frequency of the signal (float)
    
    return : The Cumulative mean normalized difference function (1D numpy array) and the estimated period in seconds (float) 

    """
    df = DiffFunction_new(x,N,t0,W,Fs)
    cmndf = CumulDiff(df)
    estimate_tau = Threshold(cmndf)
    estimate_tau = Interpol(cmndf,Fs,estimate_tau)

    return cmndf, estimate_tau
    

def FundFreqTime(x,t,N,Fs,fmin,fmax):
    """YIN Method on a time axis
    
    param x : signal to be analyzed (1D numpy array)
    param t : time axis of the signal (1D numpy array)
    param N : lenght of the signal (int)
    param Fs : Sampling frequency of the signal (float) 
    param W : lenght of the window (int)
    para fmin : The minimum frequency of the signal (float)
    para fmax : The maximum frequency of the signal (float)
    
    return : The The Cumulative mean normalized difference function (1D numpy array) and the estimated period in seconds (float) 

    """  
    tau_max = int(Fs/fmin)
    W = tau_max
    L = int(N/W)  
    term3 = np.zeros(N,dtype=float)
    for n in range(N):
        indbeg = np.max([[0],[n-int(W/2)]])
        indend = np.min([[N],[n+int(W/2)]])
        term3[n] = np.sum(x[indbeg:indend]**2)
    
    # the calculation for frame l #
        # ----------------------- #
    L = int(N/W)                        # number of loops
    period_in_time = np.ones(L)
    for l in range(L):
        indbeg = l*W                    # index of bginning of the current frame
        indend = (l+1)*W                # index of end of the current frame
        xtmp = x[indbeg:indend]         # extraction of corresponding x values
        # estimation of the (not shifted) ACF #
        # ----------------------------------- #
        Rxxtmp = np.real(ifft(np.abs(fft(xtmp))**2))
        # dtmp corresponds to d_t(\tau) in step 2 #
        # --------------------------------------- #
        df = np.zeros(int(W/2),dtype=float)
        for n in range(int(W/2)):
            df[n] = Rxxtmp[0]-2*Rxxtmp[n]+term3[l*W+int(W/2)]
            
        if np.sum(df) != 0:
            cmndf = CumulDiff(df)
            estimate_tau = Threshold(cmndf)
        
            estimate_tau = Interpol(cmndf,Fs,estimate_tau)
        else: estimate_tau = 1
        period_in_time[l] = estimate_tau
        
    time_t = [W*i/Fs for i in range(L)]
    return time_t, period_in_time
 
def FundFreqTime_new(x,t,N,Fs,fmin,fmax):
    """YIN Method
    
    param x : signal to be analyzed (1D numpy array)
    param t : time axis of the signal (1D numpy array)
    param N : lenght of the signal (int)
    param Fs : Sampling frequency of the signal (float) 
    param W : lenght of the window (int)
    para fmin : The minimum frequency of the signal (float)
    para fmax : The maximum frequency of the signal (float)
    
    return : The The Cumulative mean normalized difference function (1D numpy array) and the estimated period in seconds (float) 

    """  
    tau_max = int(Fs/fmin)
    W = tau_max
    L = int(N/W)  
    b = np.ones(W)
    term3 = lfilter(b,[1.],x**2)    
    term3 = shift_signal_in_frequency_domain(term3,-W+1)  
    
    # the calculation for frame l #
        # ----------------------- #
    L = int(N/W)                        # number of loops
    period_in_time = np.ones(L)
    for l in range(L):
        indbeg = l*W                    # index of bginning of the current frame
        indend = (l+1)*W                # index of end of the current frame
        xtmp = x[indbeg:indend]         # extraction of corresponding x values
        # estimation of the (not shifted) ACF #
        # ----------------------------------- #
        Rxxtmp = np.real(ifft(np.abs(fft(xtmp))**2))
        # dtmp corresponds to d_t(\tau) in step 2 #
        # --------------------------------------- #
        df = np.zeros(int(W/2),dtype=float)
        for n in range(int(W/2)):
            df[n] = Rxxtmp[0]-2*Rxxtmp[n]+term3[l*W+int(W/2)]
        df = df-df[0]
        if np.sum(df) != 0:
            cmndf = CumulDiff(df)
            estimate_tau = Threshold(cmndf)
        
            estimate_tau = Interpol(df,Fs,estimate_tau)      #NOTE : here, it's the difference function, not the cmndf!
        else: estimate_tau = 1
        period_in_time[l] = estimate_tau
        
    time_t = [W*i/Fs for i in range(L)]
    return time_t, period_in_time 
       
def Signal(Fs=44080,T=4):
# general parameters #
# ------------------ #

    N = int(Fs*T)                       # number of points
    t = np.arange(N)/Fs                 # time axis

    # parameters of the sine #
    # ---------------------- #
    A,B = 2, 4.6
    F1, F2 = 300, 200
    Omega1, Omega2 = 2*np.pi*F1, 2*np.pi*F2
    phi1, phi2 = 2*np.pi*np.random.rand(2)
    x1 = A*np.sin(Omega1*t+phi1)
    x2 = B*np.sin(Omega2*t+phi2)
    x = np.zeros(N,dtype=float)
    x[:int(N/2)] = x1[:int(N/2)]
    x[int(N/2):] = x2[int(N/2):]
    x = x1+x2
    N = len(x)
    t = np.arange(N)/Fs 
    return Fs,t,x

def TimeFrequencyNumerical():

    Fs,t,x = Signal(Fs=44080,T=4)
     
    fmin = 10
    fmax = 700
    N = len(x)
    
    start_time = time.time()   
    
    time_for_period, period_in_time = FundFreqTime(x,t,N,Fs,fmin,fmax)
    #cmndf,estimate_tau = YIN(x,N,t0,W,Fs,fmin,fmax)
    print("--- %s seconds ---" % (time.time() - start_time))
    return time_for_period, period_in_time

def TimeFrequencySignal(path='',file='AUD Euphonium 1 embouchure alliance E3A.wav',fmin=20,fmax=500):

    start_time = time.time()
    Fs, data = wavfile.read(path+file)

    x = data
    x = x - np.mean(x)
    x = x/np.max(np.abs(x))
    
    N = len(x)
    t = [i/Fs for i in range(N)]

    time_for_period, period_in_time = FundFreqTime_new(x,t,N,Fs,fmin,fmax)

    
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.figure()
    plt.plot(time_for_period, 1/period_in_time)
    plt.xlim(time_for_period[0],time_for_period[-1])
    plt.grid()
    plt.ylabel('Fréquence en Hz')
    plt.xlabel('Temps en secondes')

def Analyse(path = '',file = 'AUD Euphonium 1 embouchure alliance E3A.wav'):
    def resetSliders(event):    
        slder1.reset()
        slder2.reset()
            
        
    def val_update(val):
        t0 = slder1.val
        tW = slder2.val
        Nt0 = int(t0*Fs)
        W = int(tW*Fs)
        p1.set_xdata(t[Nt0:Nt0+W])
        p1.set_ydata(x[Nt0:Nt0+W])
        p2.set_xdata(t[Nt0+W:])
        p2.set_ydata(x[Nt0+W:])
        p3.set_xdata(t[:Nt0])
        p3.set_ydata(x[:Nt0])
        
        cmndf,est_tau = YIN(x,N,t0,W,Fs,fmin,fmax)
        N_est_tau = int(est_tau*Fs)
        tau = [i/Fs for i in range(len(cmndf))]
        p4.set_xdata(tau)
        p4.set_ydata(cmndf)
        p5.set_xdata(N_est_tau)
        ax2.set_xlim(tau[0],tau[-1])
        ax2.set_ylim(np.min(cmndf),np.max(cmndf))
        plt.draw()
        txt.set_text(r'$\tau_1 = {:.4f}$'.format(est_tau[0]))
    
    Fs, data = wavfile.read(path+file)
    
    x = data
    x = x - np.mean(x)
    x = x/np.max(np.abs(x))
    fmin = 20
    fmax = 500
    
    N = len(x)
    t = [i/Fs for i in range(N)]
    t0 = 0
    Nt0 = int(t0*Fs)
    tau_max = 2/fmin
    tW = tau_max
    W = int(tW*Fs)
    
    #Plot of the temporal signal
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    plt.subplots_adjust(left=0.1,bottom=0.35)
    p1, = ax1.plot(t[Nt0:Nt0+W],x[Nt0:Nt0+W],color='b')
    p2, = ax1.plot(t[Nt0+W:],x[Nt0+W:],color='b',alpha=0.5)
    p3, = ax1.plot(t[:Nt0],x[:Nt0],color='b',alpha=0.5)
    
    
    #Plot of the YIN method
    ax2 = fig1.add_subplot(122)
    cmndf,est_tau = YIN(x,N,t0,W,Fs,fmin,fmax)
    N_est_tau = int(est_tau*Fs)
    tau = [i/Fs for i in range(len(cmndf))]
    p5 = ax2.axvline(N_est_tau,color='r')
    p4, = ax2.plot(tau,cmndf,label='YIN')
    plt.ylim(np.min(cmndf),np.max(cmndf))
    ax2.set_xlim(tau[0],tau[-1])
    ax2.set_ylim(np.min(cmndf),np.max(cmndf))
    txt = plt.text(0.5,0.5,r'$\tau_1 = {:.4f}$'.format(est_tau[0]),horizontalalignment='right',
         verticalalignment='top', transform = ax2.transAxes, fontsize=14, color='r',position=(0.95,0.95))
    #Creation of sliders
    axSlider1 = plt.axes([0.1,0.2,0.8,0.05])
    slder1 = Slider(axSlider1,'t0 (in s)',valmin=t[0] ,valmax=t[-1],valinit=t0)
    
    axSlider2 = plt.axes([0.1,0.1,0.8,0.05])
    slder2 = Slider(axSlider2,'Window (in s)',valmin=t[0] ,valmax=t[-1]/4,valinit=tW)
    
    axReset = plt.axes([0.1,0.25,0.05,0.05])
    reset = Button(axReset, 'Reset')
    
    
    #Plot layouts 
    ax1.set_title('Signal temporel')
    ax1.set_xlabel('t (en s)')
    ax1.set_ylabel('Amplitude')
    
    ax2.set_title('Méthode YIN')
    ax2.set_xlabel(r'$\tau$ (en s)')
    ax1.set_ylabel('Amplitude')
    
    slder1.on_changed(val_update)
    slder2.on_changed(val_update)
    reset.on_clicked(resetSliders)    


