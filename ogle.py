from uncertainties import ufloat
from uncertainties.umath import *
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.stats import LombScargle
from scipy import signal
import warnings
warnings.filterwarnings("ignore")
import scipy.optimize
# from lmfit.models import GaussianModel
import glob
from astropy.table import Table,join,vstack,unique
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def getIV(num,cross,printall=False,stack=False,both=True,plot=False,size=4,figsize=(8,4),zooms=False,mult=(3,40),offset=0):
    '''Uses table (cross) to make lists of I band and V band tables
    mult: tuple of multiples of orbital period to show
    offset: offset from beginning of light curve in days to use for zooms
    TO DO: add errors to plots'''
    #row of cross table using source number passed in
    crow = cross[cross['src_n']==num]
    #get RA and Dec to use in title
    ra,dec = crow['RA_OGLE'][0],crow['DEC_OGLE'][0]
    #get orbital period
    if crow['Porb'].mask[0]: orb_bool=False
    else:
        orb_bool=True
        orb = crow['Porb'][0]
    #list of I band tables (length <=3)
    iband = []
    for i in ['OII I','OIII I']:
        #doesn't work for OIV I since none are masked
        if not crow[i].mask[0]: #if there's a file (not masked)
            #read in table as temporary tab
            tab = Table.read(crow[i][0],format='ascii',names=['MJD-50000','I mag','I mag err'])
            #add tab to list of I band tables
            iband.append(tab)
        else: 
            if printall: print('no file for '+I)
    #append OIV I band
    tab = Table.read(crow['OIV I'][0],format='ascii',names=['MJD-50000','I mag','I mag err'])
    iband.append(tab)
    
    #repeat for V band if both
    if both: 
        vband = []
        for v in ['OII V','OIII V','OIV V']:
            if not crow[v].mask[0]: #if there's a file (not masked)
                #read in table as temporary tab
                tab = Table.read(crow[v][0],format='ascii',names=['MJD-50000','V mag','V mag err'])
                #add tab to list of I band tables
                vband.append(tab)
        #return lists of I band and V band tables

    if plot:
        #stack for ease
        ib = vstack(iband)
        vb = vstack(vband)

        #plot both full LC and two levels of zoom-in
        if zooms:
            #for now sharey but better to give different bounds for zooms 
            fig,[ax,ax1,ax2] = plt.subplots(3,1,figsize=figsize)         
        #plot both LCs
        else: fig,ax = plt.subplots(1,1,figsize=figsize)
        maxmag = 0
        minmag = np.inf
        ax.scatter(ib['MJD-50000'],ib['I mag'],color='#CF6275',s=size)
        maxmag = np.max(ib['I mag'])
        minmag = np.min(ib['I mag'])
        ax.scatter(vb['MJD-50000'],vb['V mag'],color='navy',s=size)
        if np.max(vb['V mag'])>maxmag: 
            maxmag = np.max(vb['V mag'])
        if np.min(vb['V mag'])<minmag: 
            minmag = np.min(vb['V mag'])
        ax.set_xlabel('MJD-50000')
        ax.set_ylabel('OGLE mag')
        ax.set_ylim(maxmag+.05,minmag-.05)
        ax.set_title('src_n: '+str(num)+' RA: '+str(ra)+' Dec: '+str(dec))

        if zooms: #for now just plots I band
            #ax1 zoom is hundreds of days
            #find median time spacing between points
            samp = np.median(ib['MJD-50000'][1:] - ib['MJD-50000'][:-1])
            if orb_bool:
                inds1 = int(mult[1]*orb/samp)
                inds2 = int(mult[0]*orb/samp)
                start = int(offset/samp)
                zi1 = ib[start:start+inds1]
                zi2 = ib[start:start+inds2]
                ax1.scatter(zi1['MJD-50000'],zi1['I mag'],color='#CF6275',s=size+4)
                ax2.scatter(zi2['MJD-50000'],zi2['I mag'],color='#CF6275',s=size+4)
                #find min and max for each and adjust y lim
                max1,min1 = np.max(zi1['I mag']),np.min(zi1['I mag'])
                max2,min2 = np.max(zi2['I mag']),np.min(zi2['I mag'])
                ax1.set_ylim(max1+.02,min1-.02)
                ax2.set_ylim(max2+.02,min2-.02)               
                print('orbital period:',orb)

            else: #if no known orbital period
                #TO DO add in offset use here
                inds1 = int(1000/samp)
                inds2 = int(100/samp)
                zi1 = ib[:inds1]
                zi2 = ib[:inds2]
                ax1.scatter(zi1['MJD-50000'],zi1['I mag'],color='#CF6275',s=size+4)
                ax2.scatter(zi2['MJD-50000'],zi2['I mag'],color='#CF6275',s=size+4)
                #find min and max for each and adjust y lim
                max1,min1 = np.max(zi1['I mag']),np.min(zi1['I mag'])
                max2,min2 = np.max(zi2['I mag']),np.min(zi2['I mag'])
                ax1.set_ylim(max1+.02,min1-.02)
                ax2.set_ylim(max2+.02,min2-.02)     
    if stack and both: return vstack(iband),vstack(vband)
    elif both: return iband,vband
    elif stack: return vstack(iband)
    else: return iband
    
def colormag(iband,vband,figsize=(5,4)):
    '''Interpolates I band data at times of V band and then plots color-mag with best fit and corr coeff.
    Now assumes iband and vband are single tables, but can add option to vstack in function if needed.'''
    #interpolate I band
    i_interp = np.interp(vband['MJD-50000'],iband['MJD-50000'],iband['I mag'])
    
    plt.figure(figsize=figsize)
    #plot Iint vs. V-I
    plt.scatter(vband['V mag']-i_interp,i_interp,color='black')
    #flip y-axis such that positive corr on plot is redder when brighter
    maxi,mini = np.max(i_interp),np.min(i_interp)
    plt.ylim(maxi+.04,mini-.04)
    plt.ylabel(r'$\mathrm{I_{int}}$',fontsize=14)
    plt.xlabel(r'$\mathrm{V - I_{int}}$',fontsize=14)
    #print correlation corr with interpolated I and V-I and then V and V-I
    print('I and V-I correlation:',np.corrcoef(vband['V mag']-i_interp,i_interp)[1][0])
    print('V and V-I correlation:',np.corrcoef(vband['V mag']-i_interp,vband['V mag'])[1][0]) 
    return


#-----------------------------------------------------------------PERIODOGRAMS--------------------------------------------------------------


def knownorb(itab,orb,lower=10,upper=10,window=11,cutdata=False,cut1=0,cut2=500,plotdet=False,figsize=(12,4),plotpd=True):
    '''Use known orbital period (or estimate) to inform detrending and periodogram.
    lower and upper subtracted/added onto orb to give periodogram bounds
    small detrending window default
    Can run on all data (stacked tab) or pass in list
    cutdata: run periodogram on given inds
        cut1: lower index of itab or itab[0] to use for periodogram
        cut2: upper index of itab or itab[0] to use for periodogram
    plotdet: plot detrended I mag used for periodogram
    
    TO DO: decide if separate pdgrams or on one plot better
            gridspec so that LC has more space
    TO DO: make it simple/add func to loop thru det windows etc. and plot results'''
    #check if list --> one periodogram for each element
    if len(itab)<=3:
        best_ps = [] #list of best periods
        #loop thru tables in itab list and run periodogram on each
        if plotpd and not plotdet: fig,ax = plt.subplots(1,len(itab),figsize=figsize)
        #plot multiple periodograms and
        elif plotdet: fig,ax = plt.subplots(1,len(itab)+1,figsize=figsize)
        c=0
        minmag,maxmag = np.inf,0
        for i in itab:
            #to not modify itab?
            tab = i
            #get pdgram results but don't plot within function
            #detrends within
            freq, power, best_p = periodogram(tab,det=True,more=True,minp=orb-lower,maxp=orb+upper,plot=False,dodetrend=True,window=window)
            if plotpd:
                ax[c].plot(1/freq,power,color='black')
                #text with best period
                minp,maxp = orb-lower,orb+upper
                ax[c].text(minp+(maxp-minp)/2,0.8*np.max(power),f'{best_p:2f}')
            #plot all detrended data in last ax
            if plotdet:
                ax[-1].scatter(tab['MJD-50000'],tab['I detrend'],color='#CF6275',s=2)
                if np.min(tab['I detrend'])<minmag: minmag = np.min(tab['I detrend'])
                if np.max(tab['I detrend'])>maxmag: maxmag = np.max(tab['I detrend'])
            c+=1
            best_ps.append(best_p)
        #flip y-axis of detrended data
        if plotdet: ax[-1].set_ylim(maxmag+0.05,minmag-0.05)
        return best_ps
    else:
        #one periodogram for table passed in 
        #TO DO give option to further cut
        if plotpd and not plotdet: fig,ax = plt.subplots(1,1,figsize=figsize)
        #plot multiple periodograms and detrended LC
        elif plotdet: fig,(ax,axd) = plt.subplots(2,1,figsize=figsize)
        #only use data in cut1,cut2 indices
        if cutdata:
            tab = itab[cut1:cut2]
        else: tab = itab
        freq, power, best_p = periodogram(tab,det=True,more=True,minp=orb-lower,maxp=orb+upper,plot=False,dodetrend=True,window=window)
        #print(best_p)
        if plotpd: ax.plot(1/freq,power,color='black')
        #text with best period
        minp,maxp = orb-lower,orb+upper
        if plotpd: ax.text(minp+(maxp-minp)/2,0.8*np.max(power),f'{best_p:2f}')
        if plotdet:
            #plot detrended data
            axd.scatter(tab['MJD-50000'],tab['I detrend'],color='#CF6275',s=2)
            minmag = np.min(tab['I detrend'])
            maxmag: maxmag = np.max(tab['I detrend'])
            #flip y-axis of detrended data
            axd.set_ylim(maxmag+0.05,minmag-0.05)
        return best_p
        
def window_loop(itab,orb,lower=10,upper=10,window=[11,31,51],cutdata=False,cut1=0,cut2=500,plotdet=False,figsize=(6,4),plotpd=False,plotloop=True):
    '''Use knownorb without plotting to try many window, cuts, etc. to see effect on best period
    TO DO: add options for other loops e.g. where data is cut, or upper and lower bounds'''
    bps = []
    for w in window:
        best_p = knownorb(itab,orb,lower=lower,upper=upper,window=w,cutdata=cutdata,cut1=cut1,cut2=cut2,plotdet=plotdet,plotpd=plotpd)
        bps.append(best_p.value)
    
    if plotloop:
        plt.figure(figsize=figsize)
        plt.scatter(window,bps,color='black')
        plt.ylabel('Best Period (days)')
        plt.xlabel('Detrending Window')

    return bps


def pltphase(tab,best_p,freq,power,figsize=(6,4),inpd=True,inwin=False,wins=[],winpds=[],title='Largest Trend ',
             size=2,ctime=False,cmap='magma',inloc='lower right',plotdet=False,avgph=True,mids=[],avgs=[],avgcolor='black',save=False,srcnum=7):
    '''Plot phase-folded data, with option to inset periodogram or window loop results'''
    fig,ax = plt.subplots(figsize=figsize)
    if inpd or inwin:
        axins = inset_axes(ax, width=figsize[0]/4, height=figsize[1]/4,loc=inloc,borderpad=0.5)
        #move x-axis to top
        if 'lower' in inloc: axins.xaxis.tick_top()
        if inpd: axins.plot(1/freq,power,color='black')
        elif inwin: 
            axins.scatter(wins,winpds,color='black')
#             axins.axvline() to do: add line for known orbital period
    if plotdet: imag = tab['I detrend']
    else: imag = tab['I mag']
    #color based on day to show evolution
    if ctime: 
        im = ax.scatter(tab['MJD-50000']%best_p,imag,c=tab['MJD-50000'],s=size,cmap=cmap)
        cax = inset_axes(ax,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0,
                   )
        fig.colorbar(im,cax=cax,label='MJD-50000')
    else: ax.scatter(tab['MJD-50000']%best_p,imag,color='black',s=size)
    ax.set_title(title+f'{best_p:1f}'+' days')
    ax.set_xlabel('Phase (days)')
    ax.set_ylabel('I mag')
    #flip y-axis
    maxmag,minmag = np.max(imag),np.min(imag)
    #tested manually putting in lims b/c of many outliers
    ax.set_ylim(maxmag+.04,minmag-.04)
        
    #plot average in phase bins
    if avgph:
        ax.plot(mids,avgs,color=avgcolor)
    if save:
        plt.savefig('Figs/'+title[:-1]+str(srcnum)+'.png',dpi=200,bbox_inches='tight')
        print('saving as '+'Figs/'+title[:-1]+str(srcnum)+'.png')
    return

def autopd(tab,orb,printall=True,plotpd=False,plotphase=False,figsize=(3,2),figsizebig=(6,4),plotloop=True,
           cutlc=True,numcuts=10,ctime=False,cmap='magma',inloc='lower right',orb_bounds=(20,20),plotdet=True,pbins=20,saveall=False,srcnum=7,medlow=4):
    '''Look for periodicity on three scales; detrending informed by results of each iteration
    Loop through reasonable range of detrending windows each time
    Pass in single tab (stacked or one og at a time to compare)
    Be sure to specify srcnum if saving figs
    TO DO 
    figure out better way to choose windows based on trend (use sampling)
    add power of window results as another dimension of window plot
    option to plot detrended when detrended used for periodogram
    other options for different scales of search and detrending
    
    '''
    #find total time in LC
    tot_days = tab['MJD-50000'][-1] - tab['MJD-50000'][0]
    #get best period on whole LC without detrending
    tfreq,tpower,tbp = periodogram(tab,more=True,minp=tot_days/5,maxp=tot_days/2,plot=plotpd,figsize=figsize)
    #print best period and power of best period
    if printall: print('Largest best period: ',tbp,'with power of: ',np.max(tpower))
    #plot phase-folded data using best period
    if plotphase:
        plt.close() #get rid of earlier pdgram plot and inset instead
        pltphase(tab,tbp,tfreq,tpower,figsize=figsizebig,inpd=plotpd,ctime=ctime,cmap=cmap,inloc=inloc,save=saveall,srcnum=srcnum)
        
    #run periodogram with reasonable bounds both with and without detrending of first trend above 
    #boolean for plotting periodogram using function --> only if plotpd but not plotphase
    per_bool = not plotphase and plotpd
    #medlow specifies multiples of orbital period as lower bound of medium trend search
    freq2,power2,bp2 = periodogram(tab,more=True,minp=orb*medlow,maxp=tbp/2,plot=per_bool,figsize=figsize)
    if printall: print('Medium best period (without detrending): ',bp2)

    #several detrendings based on result of largest-scale trend
    #TO DO: better range of windows
    windows = np.arange(29,101,16)
    win_bool = plotloop and not plotphase
    bps = window_loop(tab,orb,lower=(-medlow*orb)+orb,upper=tbp/2,window=windows,plotloop=win_bool,figsize=figsize)
    if printall: print('Window results for medium: mean period ',np.mean(bps),' stdev: ',np.std(bps))

    if plotphase:
        #plot phase-fold (and periodogram inset if plotpd) of non-detrended 
        pltphase(tab,bp2,freq2,power2,figsize=figsizebig,inpd=plotpd,title='Medium Trend ',size=3,ctime=ctime,cmap=cmap,inloc=inloc)
        #plot phase fold with median best period from window search
        #pass in same power but not used since no periodogram inset
        #inset window loop 
        #may plot detrended
        detrend(tab,window=81)
        pltphase(tab,np.mean(bps),freq2,power2,figsize=figsizebig,inpd=False,inwin=True,wins=windows,winpds=bps,
                 title='Mean Medium Window Trend ',size=3,ctime=ctime,cmap=cmap,inloc=inloc,save=saveall,srcnum=srcnum,plotdet=plotdet)
   
    #last trend search: orbital or predicted orbital (pass in orb even if no known orbital period)
    #search with and without detrending; option to cut up LC (bool cutlc) and search in each part (int numcuts)
    
    if cutlc:
        #first: pdgram on chunks without detrending
        totinds = len(tab)
        #use total points to break up into chunks
        cinds = np.arange(0,totinds,int(totinds/numcuts))
        #loop thru cinds 
        cfreqs,cpows,cbps = [],[],[]
        if plotpd:
            fig,ax = plt.subplots(1,1,figsize=figsizebig)
        #TO DO: distinguish btwn pdgrams in time
        maxpow = 0
        for i in range(1,len(cinds)):
            #cut tab
            ctab = tab[cinds[i-1]:cinds[i]]
            cfreq,cpower,cbp = periodogram(ctab,more=True,minp=orb-orb_bounds[0],maxp=orb+orb_bounds[1],plot=per_bool,figsize=figsize)
            #plot all  periodograms in one plot
            if plotpd: 
                ax.plot(1/cfreq,cpower,color='black')
            cfreqs.append(cfreq)
            cpows.append(cpower)
            cbps.append(cbp)
            if np.max(cpower)>maxpow: maxpow = np.max(cpower)
        if plotpd: 
            ax.set_ylabel('Power')
            ax.set_xlabel('Period (days)')
            ax.axvline(orb,color='red',alpha=0.4)
            #inset best period vs. chunk number
            axins = inset_axes(ax, width=figsizebig[0]/4, height=figsizebig[1]/4,loc='upper left',borderpad=0.5)
            axins.scatter(np.arange(numcuts),cbps,color='black')
            ax.set_ylim(0,maxpow+maxpow/4)
            axins.yaxis.tick_right()
            ax.set_title('Search for Orbital Period on Chunks of LC')
            if saveall: plt.savefig('Figs/orbchunks'+str(srcnum)+'.png',dpi=200,bbox_inches='tight')
    #search on whole LC without detrending (before was else statement but including always)
#     else:
    ofreq,opower,obp = periodogram(tab,more=True,minp=orb-orb_bounds[0],maxp=orb+orb_bounds[1],plot=per_bool,figsize=figsize)
    if printall: print('Small (orbital) best period (without detrending): ',obp)
    if plotphase:
        #plot phase-fold (and periodogram inset if plotpd) of non-detrended 
        #if plotdet: plot phase-folded detrended even though pdgram on non-det
        if plotdet:
            #to do: allow for input of this window; determine more robustly
            detrend(tab,window=71)
        mid,avgs = meanphase(tab,obp,pbins=pbins,det=plotdet)
        pltphase(tab,obp,ofreq,opower,figsize=figsizebig,inpd=plotpd,title='Smallest Trend Without Detrending ',size=2,ctime=ctime,cmap=cmap,
                    inloc=inloc,plotdet=plotdet,avgph=True,mids=mid,avgs=avgs,save=saveall,srcnum=srcnum)
    #whether or not cutlc used, try various detrendings on full LC and plot phase-fold with mean (or known period?) and window plot
    windows = np.arange(7,81,16)
    obps = window_loop(tab,orb,lower=orb-orb_bounds[0],upper=orb+orb_bounds[1],window=windows,plotloop=win_bool,figsize=figsize)
    if printall: print('Window results for small: mean period ',np.mean(obps),' stdev: ',np.std(obps))

    if plotphase:
        #plot phase fold with mean best period from window search
        #pass in same power but not used since no periodogram inset
        #inset window loop 
        if plotdet:
            #to do: allow for input of this window; determine more robustly
            detrend(tab,window=81)
        omid,oavgs = meanphase(tab,np.mean(obps),pbins=pbins,det=plotdet)
        pltphase(tab,np.mean(obps),freq2,power2,figsize=figsizebig,inpd=False,inwin=True,wins=windows,winpds=obps,
                 title='Mean Small Window Trend ',size=3,ctime=ctime,cmap=cmap,inloc=inloc,plotdet=plotdet,avgph=True,
                 mids=omid,avgs=oavgs,save=saveall,srcnum=srcnum)
        
            
            
    return 
    #return tbp,bp2,np.mean(bps),obp

def meanphase(tab,pd,pbins=20,det=False):
    fr = tab.to_pandas() #don't modify tab
    fr['phase'] = (fr['MJD-50000']%pd)
    fr = fr.sort_values(by='phase',ascending=True)
    #use detrended or regular imag
    if det: imag = fr['I detrend']
    else: imag = fr['I mag']
    #find average count rate in each phase bin
        
    #other method with just loop length of number of phase bins
    #for now just one to do all the necessary filtering
    avgs = [] #list of average count rate in each phase bin
    endb = np.arange(pd/pbins,pd+pd/pbins,pd/pbins)
    for p in endb:
        #phase in temporary df is less than phase in endb and more than the previous one
        tempfr = fr[fr['phase']<=p]
        tempfr = tempfr[tempfr['phase']>p-pd/pbins]
        avgs.append(np.mean(imag))
    endb2 = np.concatenate([np.array([0]),endb])
    #middle of phase bins
    mid = (endb2[1:]+endb2[:-1])/2
    return mid,avgs


def detrend(tab,window=201,printall=False,plot=False):
    Imag = tab['I mag']
    if printall: print('Smooth (window = ', window, ') and detrend data...')
    Ismooth = signal.savgol_filter(Imag, window, 1)
    Imean = np.mean(Imag)
    if printall: print('Average I band magnitude', Imean)
    tab['I detrend'] = Imag-Ismooth  + Imag.mean()

    if printall: print('min:',i26['I detrend'].min(),'max:',i26['I detrend'].max())
    if plot:
        plt.scatter(tab['MJD-50000'],tab['I mag'],color='black',label='original')
        plt.scatter(tab['MJD-50000'],tab['I detrend'],color='darkseagreen',label='detrended')
        plt.legend()
        
def periodogram(tab,det=False,more=False,minp=5,maxp=30,bayes=False,sub=False,figsize=(4,3),plot=True,dodetrend=False,window=11):
    '''Perform and plot single LS periodogram.
    Two different return options.'''
    
    t = tab['MJD-50000']
    if dodetrend:
        #decide whether to actually modify tab or create copy just for periodogram
        detrend(tab,window=window)
    if det: y = tab['I detrend']
    else: y = tab['I mag']
    dy = tab['I mag err']
    minf = 1./maxp
    maxf = 1./minp
    ls = LombScargle(t, y)
    freq, power = ls.autopower(normalization='standard',
                           minimum_frequency=minf,
                           maximum_frequency=maxf,
                           samples_per_peak=10)
    if bayes: power = np.exp(power)
        
    best_freq = freq[np.argmax(power)]

    if plot:
        fig = plt.figure(figsize=figsize)
        plt.plot(1/freq,power,color='black')
        plt.xlabel('Period',fontsize=14)
        plt.ylabel('Power',fontsize=14)
        #put text with best period
        plt.text(minp+(maxp-minp)/2,0.8*np.max(power),f'{1/best_freq:2f}')
    if more:
        return freq, power, 1/best_freq
    else:
        return 1/best_freq