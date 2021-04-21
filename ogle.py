from uncertainties import ufloat,unumpy
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
import wotan
import scipy.stats as st


def sf(name,dpi=200,path='Figs/'):
    plt.savefig(path+name+'.png',dpi=dpi,bbox_inches='tight')

#initial LC, then color-mag, then a bunch of periodogram functions

def getIV(num,cross,printall=False,stack=False,both=True,plot=False,size=4,figsize=(8,4),zooms=False,mult=(3,40),offset=0,save=False,file='',radec=True,mlist=['OII I','OIII I'],calib=False):
    '''Uses table (cross) to make lists of I band and V band tables
    mult: tuple of multiples of orbital period to show
    offset: offset from beginning of light curve in days to use for zooms
    mlist: list of file names with masking in them; different for Part 2
    TO DO: add errors to plots'''
    #row of cross table using source number passed in
    crow = cross[cross['src_n']==num]
    #get RA and Dec to use in title
    if radec: ra,dec = crow['RA_OGLE'][0],crow['DEC_OGLE'][0]
    #get orbital period
    if crow['Porb'].mask[0]: orb_bool=False
    else:
        orb_bool=True
        orb = crow['Porb'][0]
    #list of I band tables (length <=3)
    iband = []
    for i in mlist:
        #doesn't work for OIV I since none are masked
        if not crow[i].mask[0]: #if there's a file (not masked)
            #read in table as temporary tab
            tab = Table.read(crow[i][0],format='ascii',names=['MJD-50000','I mag','I mag err'])
            #add tab to list of I band tables
            if len(tab)>0: iband.append(tab)
            else: print(f'empty file for {i}')
        else: 
            if printall: print('no file for '+i)
    #append OIV I band
    if len(mlist)<3:
        tab = Table.read(crow['OIV I'][0],format='ascii',names=['MJD-50000','I mag','I mag err'])
        if len(tab)>0: iband.append(tab)
        else: print(f'empty file for OIV I')
    
    #repeat for V band if both
    if both: 
        vband = []
        for v in ['OII V','OIII V','OIV V']:
            if not crow[v].mask[0]: #if there's a file (not masked)
                #read in table as temporary tab
                tab = Table.read(crow[v][0],format='ascii',names=['MJD-50000','V mag','V mag err'])
                #add tab to list of I band tables
                if len(tab)>0: vband.append(tab)
                else: print(f'empty file for {v}')
        #return lists of I band and V band tables
    #compensate for uncalibrated data by setting epochs to a common median, which is the overall median
    #V band unchanged
    if calib:
        itemp = vstack(iband)
        med = np.median(itemp['I mag'])
        for i in iband:
            #calculate current median
            cmed = np.median(i['I mag'])
            #difference between current and target median
            dmeds = med-cmed
            #add difference to all points
            i['I mag'] += dmeds
    if plot:
        #stack for ease
        ib = vstack(iband)
        if both: vb = vstack(vband)

        #plot both full LC and two levels of zoom-in
        if zooms:
            #for now sharey but better to give different bounds for zooms 
            fig,[ax,ax1,ax2] = plt.subplots(3,1,figsize=figsize)         
        #plot both LCs
        else: fig,ax = plt.subplots(1,1,figsize=figsize)
        maxmag = 0
        minmag = np.inf
        ax.scatter(ib['MJD-50000'],ib['I mag'],color='#CF6275',s=size,label='I band')
        maxmag = np.max(ib['I mag'])
        minmag = np.min(ib['I mag'])
        if both:
            ax.scatter(vb['MJD-50000'],vb['V mag'],color='navy',s=size,label='V band')
            if np.max(vb['V mag'])>maxmag: 
                maxmag = np.max(vb['V mag'])
            if np.min(vb['V mag'])<minmag: 
                minmag = np.min(vb['V mag'])
        ax.set_xlabel('MJD-50000',fontsize=14)
        ax.set_ylabel('OGLE mag',fontsize=14)
        ax.set_ylim(maxmag+.05,minmag-.05)
        if radec: ax.set_title('Source #'+str(num)+' RA: '+str(ra)+' Dec: '+str(dec))
        else: ax.set_title('Source #'+str(num))
        ax.legend()
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
    if save: plt.savefig(file+'.png',dpi=200,bbox_inches='tight')
    if stack and both: return vstack(iband),vstack(vband)
    elif both: return iband,vband
    elif stack: return vstack(iband)
    else: return iband
    
def gallery(cross,n=0,ctime=True,color='#CF6275',cmap='viridis'): #assumes several variables defined already
    '''Make gallery of 10 sources with I mag LC and color-mag diagram
    To do: generalize; account for tables that pass in with Strings rather than floats
    colorbar?'''
    fig = plt.figure(figsize=(22,18))
    plt.subplots_adjust(wspace=.25,hspace=.25)
    c = 1
    while c < 20:
        i,v = getIV(cross['src_n'][n],cross,stack=True,both=True,plot=False,size=2)
        ax = fig.add_subplot(5,4,c)
        ax.scatter(i['MJD-50000'],i['I mag'],color=color,s=3)
        ax.set_title(str(cross['src_n'][n]))
        maxi,mini = np.max(i['I mag']),np.min(i['I mag'])
        ax.set_ylim(maxi+.015,mini-.015)
        c+=1
        n+=1
        #color-mag
        ax1 = fig.add_subplot(5,4,c)
        i_interp = colormag(i,v,plot=False,printcorr=False)
        #scatter I vs V-I
        if ctime:
            #color represents V mag time (also where I is interpolated)
            ax1.scatter(v['V mag']-i_interp,i_interp,c=v['MJD-50000'],s=3,cmap=cmap)
            #also plot i_interp with same color on first plot
            ax.scatter(v['MJD-50000'],i_interp,c=v['MJD-50000'],s=3,cmap=cmap)
        else: ax1.scatter(v['V mag']-i_interp,i_interp,color='black',s=3)
        ax1.set_ylim(maxi+.01,mini-.01)
        c+=1
    return n
    
def colormag(iband,vband,figsize=(7,8),plot=True,printcorr=True,retint=False,ctime=True,cmap='viridis',both=True,save=False,file=''):
    '''Interpolates I band data at times of V band and then plots color-mag with best fit and corr coeff.
    Now assumes iband and vband are single tables, but can add option to vstack in function if needed.'''
    #interpolate I band
    i_interp = np.interp(vband['MJD-50000'],iband['MJD-50000'],iband['I mag'])
    
    if plot:
        if both: 
            fig,(ax,ax1) = plt.subplots(2,1,figsize=figsize,sharex=True)
            plt.subplots_adjust(hspace=0.05)
            axlist = [ax,ax1]
        else: 
            fig,ax = plt.subplots(1,1,figsize=figsize)
            axlist = [ax]
        #approximate interpolated I errors as median I band error (was using max but could be issue w/outliers)
        ie = np.ones(len(i_interp))*np.median(iband['I mag err'])
        #propagate errors to get error on V-I points
        verr = unumpy.uarray(vband['V mag'],vband['V mag err'])
        ierr = unumpy.uarray(i_interp,ie)
        v_i = verr-ierr
        #just take errors
        v_i_err = unumpy.std_devs(v_i)
        
        #plot Iint vs. V-I
        if ctime:
            im = ax.scatter(vband['V mag']-i_interp,i_interp,c=vband['MJD-50000'],cmap=cmap,zorder=10)
            #add errorbars
            ax.errorbar(vband['V mag']-i_interp,i_interp,yerr=ie,xerr=v_i_err,color='grey',zorder=0,ls='none',marker='')
            if both: 
                ax1.scatter(vband['V mag']-i_interp,vband['V mag'],c=vband['MJD-50000'],cmap=cmap,zorder=10)
                #add errorbars separately
                ax1.errorbar(vband['V mag']-i_interp,vband['V mag'],yerr=vband['V mag err'],xerr=v_i_err,color='grey',zorder=0,ls='none',marker='')
            fig.colorbar(im, ax=axlist,label='MJD-50000')        
        else: 
            ax.errorbar(vband['V mag']-i_interp,i_interp,yerr=ie,xerr=v_i_err,color='black',linestyle='none',marker='o')
            if both: ax1.errorbar(vband['V mag']-i_interp,vband['V mag'],yerr=vband['V mag err'],xerr=v_i_err,color='black',linestyle='none',marker='o')
        #flip y-axis such that positive corr on plot is redder when brighter
        maxi,mini = np.max(i_interp),np.min(i_interp)
        maxv,minv = np.max(vband['V mag']),np.min(vband['V mag'])
        
        
        ax.set_ylim(maxi+.04,mini-.04)
        if both:ax1.set_ylim(maxv+.04,minv-.04)
        
        ax.set_ylabel(r'$\mathrm{I_{int}}$',fontsize=14)
        if both: 
            ax1.set_xlabel(r'$\mathrm{V - I_{int}}$',fontsize=14)
            ax1.set_ylabel('V',fontsize=14)
        else: ax.set_xlabel(r'$\mathrm{V - I_{int}}$',fontsize=14)
    if printcorr:
        #print correlation corr with interpolated I and V-I and then V and V-I
        print('I and V-I correlation:',np.corrcoef(vband['V mag']-i_interp,i_interp)[1][0])
        print('V and V-I correlation:',np.corrcoef(vband['V mag']-i_interp,vband['V mag'])[1][0])
    if save: plt.savefig(file+'.png',dpi=200,bbox_inches='tight')
    if retint or not plot: return i_interp
    else: return


#-----------------------------------------------------------------PERIODOGRAMS--------------------------------------------------------------


def knownorb(itab,orb,lower=10,upper=10,window=11,cutdata=False,cut1=0,cut2=500,plotdet=False,figsize=(12,4),plotpd=True,samples=10,spline=False,btol=50):
    '''Use known orbital period (or estimate) to inform detrending and periodogram.
    lower and upper subtracted/added onto orb to give periodogram bounds
    small detrending window default
    Can run on all data (stacked tab) or pass in list
    cutdata: run periodogram on given inds
        cut1: lower index of itab or itab[0] to use for periodogram
        cut2: upper index of itab or itab[0] to use for periodogram
    plotdet: plot detrended I mag used for periodogram
    samples: samples per peak in periodogram
    
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
            #spline detrend
            if spline:
                splinedetrend(tab,window=window,btol=btol)
                print('using rspline for detrending')
            #detrends within if using Savitzky-Golay
            dodet = not spline
            freq, power, best_p = periodogram(tab,det=True,more=True,minp=orb-lower,maxp=orb+upper,plot=False,dodetrend=dodet,window=window,samples=samples)
            if plotpd:
                ax[c].plot(1/freq,power,color='black')
                ax[c].axvline(orb,color='darkseagreen',alpha=0.5)
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
        if spline:
            splinedetrend(tab,window=window,btol=btol)
            print('using rspline for detrending')
        #detrends within if using Savitzky-Golay
        dodet = not spline
        freq, power, best_p = periodogram(tab,det=True,more=True,minp=orb-lower,maxp=orb+upper,plot=False,dodetrend=dodet,window=window)
        #print(best_p)
        if plotpd: ax.plot(1/freq,power,color='black')
        #vertical line at established period
        ax.axvline(orb,color='darkseagreen',alpha=0.5)
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
        
def window_loop(itab,orb,lower=10,upper=10,window=[11,31,51],cutdata=False,cut1=0,cut2=500,plotdet=False,figsize=(6,4),plotpd=False,plotloop=True,spline=False,btol=50):
    '''Use knownorb without plotting to try many window, cuts, etc. to see effect on best period
    TO DO: add options for other loops e.g. where data is cut, or upper and lower bounds'''
    bps = []
    for w in window:
        best_p = knownorb(itab,orb,lower=lower,upper=upper,window=w,cutdata=cutdata,cut1=cut1,cut2=cut2,plotdet=plotdet,plotpd=plotpd,spline=spline,btol=btol)
        bps.append(best_p.value)
    
    if plotloop:
        plt.figure(figsize=figsize)
        plt.scatter(window,bps,color='black')
        plt.ylabel('Best Period (days)')
        plt.xlabel('Detrending Window')

    return bps


def pltphase(tab,best_p,freq,power,figsize=(6,4),inpd=True,inwin=False,wins=[],winpds=[],title='Largest Trend ',
             size=2,ctime=False,cmap='magma',inloc='lower right',plotdet=False,avgph=True,mids=[],avgs=[],avgcolor='darkseagreen',
             save=False,srcnum=7):
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
           cutlc=True,numcuts=10,ctime=False,cmap='magma',inloc='lower right',orb_bounds=(20,20),plotdet=True,
           pbins=20,saveall=False,srcnum=7,medlow=4,spline=False,btol=50,win=201):
    '''Look for periodicity on three scales; detrending informed by results of each iteration
    Loop through reasonable range of detrending windows each time
    Pass in single tab (stacked or one og at a time to compare)
    Be sure to specify srcnum if saving figs
    spline: use wotan rspline to detrend rather than Savitzky-Golay
    win: window for spline or S-G detrending
    
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
    #TO DO: best range of windows?
    windows = np.arange(101,401,30)
    win_bool = plotloop and not plotphase
    #follows spline bool for detrending method
    bps = window_loop(tab,orb,lower=-medlow*orb,upper=tbp/2,window=windows,plotloop=win_bool,figsize=figsize,spline=spline,btol=btol)
    if printall: print('Window results for medium: mean period ',np.mean(bps),' stdev: ',np.std(bps))

    if plotphase:
        #plot phase-fold (and periodogram inset if plotpd) of non-detrended 
        pltphase(tab,bp2,freq2,power2,figsize=figsizebig,inpd=plotpd,title='Medium Trend ',size=3,ctime=ctime,cmap=cmap,inloc=inloc)
        #plot phase fold with median best period from window search
        #pass in same power but not used since no periodogram inset
        #inset window loop 
        #may plot detrended
        if spline: splinedetrend(tab,window=win,btol=50)
            
        else: detrend(tab,window=win)
        pltphase(tab,np.mean(bps),freq2,power2,figsize=figsizebig,inpd=False,inwin=True,wins=windows,winpds=bps,
                 title='Mean Medium Window Trend ',size=3,ctime=ctime,cmap=cmap,inloc=inloc,save=saveall,srcnum=srcnum,plotdet=plotdet)
   
    #last trend search: orbital or predicted orbital (pass in orb even if no known orbital period)
    #search with and without detrending; option to cut up LC (bool cutlc) and search in each part (int numcuts)
    
    if cutlc:
        #first: pdgram on chunks without detrending
        totinds = len(tab)
        #use total points to break up into chunks
        cinds = np.arange(0,totinds+1,int(totinds/numcuts)) #+1 just necessary in case totinds divisible by numcuts
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
    ofreq,opower,obp = periodogram(tab,more=True,minp=orb-orb_bounds[0],maxp=orb+orb_bounds[1],plot=per_bool,figsize=figsize)
    if printall: print('Small (orbital) best period (without detrending): ',obp)
    if plotphase:
        #plot phase-fold (and periodogram inset if plotpd) of non-detrended 
        #if plotdet: plot phase-folded detrended even though pdgram on non-det
        if plotdet and spline: splinedetrend(tab,window=win,btol=btol)
            #to do: allow for input of this window; determine more robustly
        elif plotdet: detrend(tab,window=win)
        mid,avgs = meanphase(tab,obp,pbins=pbins,det=plotdet)
        pltphase(tab,obp,ofreq,opower,figsize=figsizebig,inpd=plotpd,title='Smallest Trend Without Detrending ',size=2,ctime=ctime,cmap=cmap,
                    inloc=inloc,plotdet=plotdet,avgph=True,mids=mid,avgs=avgs,save=saveall,srcnum=srcnum)
    #whether or not cutlc used, try various detrendings on full LC and plot phase-fold with mean (or known period?) and window plot
    windows = np.arange(81,201,16)
    obps = window_loop(tab,orb,lower=orb_bounds[0],upper=orb_bounds[1],window=windows,plotloop=win_bool,figsize=figsize,spline=spline,btol=btol)
    if printall: print('Window results for small: mean period ',np.mean(obps),' stdev: ',np.std(obps))

    if plotphase:
        #plot phase fold with mean best period from window search
        #pass in same power but not used since no periodogram inset
        #inset window loop 
        omid,oavgs = meanphase(tab,np.mean(obps),pbins=pbins,det=plotdet)
        pltphase(tab,np.mean(obps),freq2,power2,figsize=figsizebig,inpd=False,inwin=True,wins=windows,winpds=obps,
                 title='Mean Small Window Trend ',size=3,ctime=ctime,cmap=cmap,inloc=inloc,plotdet=plotdet,avgph=True,
                 mids=omid,avgs=oavgs,save=saveall,srcnum=srcnum)
                  
    return 
    #return tbp,bp2,np.mean(bps),obp

    
def meanphase(tab,pd,pbins=20,det=False,med=False,double=False,stdev=False,epoch=0,divide=False,sterr=False):
    '''Compute mean mag in phase bins of LC
    divide: phase only goes to 1 or 2
    sterr: return standard error (so divide stdev by square root of number of points in bin); stdev must also be True'''
#     if sterr: stdev == True
    fr = tab.to_pandas() #don't modify tab
    #add epoch for phase shift
    fr['MJD-50000'] += epoch
    fr['phase'] = (fr['MJD-50000']%pd)
    if divide: fr['phase'] /= pd
    fr = fr.sort_values(by='phase',ascending=True)
    #use detrended or regular imag
    if det: imag = fr['I detrend']
    else: imag = fr['I mag']
    #find average count rate in each phase bin
        
    #other method with just loop length of number of phase bins
    #for now just one to do all the necessary filtering
    avgs = [] #list of average count rate in each phase bin
    stdevs = []
    if divide: endb = np.arange(1/pbins,1+1/pbins,1/pbins)
    else: endb = np.arange(pd/pbins,pd+pd/pbins,pd/pbins)
    for p in endb:
        #phase in temporary df is less than phase in endb and more than the previous one
        tempfr = fr[fr['phase']<=p]
        if divide: tempfr = tempfr[tempfr['phase']>p-1/pbins]
        else: tempfr = tempfr[tempfr['phase']>p-pd/pbins]
        if det: imagt = tempfr['I detrend']
        else: imagt = tempfr['I mag']
        #use median instead
        if med:avgs.append(np.median(imagt))
        else: avgs.append(np.mean(imagt))
        #save standard deviation within each bin
        if stdev and sterr: 
        #if sterr, divide stdev by square root of number of points per bin
            denom = np.sqrt(len(tempfr)) 
            stdevs.append(np.std(imagt)/denom)
        elif stdev: #stdev but not standard error
            stdevs.append(np.std(imagt))
    endb2 = np.concatenate([np.array([0]),endb])
    #middle of phase bins
    mid = (endb2[1:]+endb2[:-1])/2
    #mids, means, and stdevs for two phases
    if stdev and double and divide: return np.concatenate([mid,1+mid]),np.concatenate([avgs,avgs]),np.concatenate([stdevs,stdevs])
    elif stdev and double: return np.concatenate([mid,pd+mid]),np.concatenate([avgs,avgs]),np.concatenate([stdevs,stdevs])
    #stdev but one phase
    elif stdev: return mid,avgs,stdevs
    #mids, meansfor two phases
    elif double and divide: return np.concatenate([mid,1+mid]),np.concatenate([avgs,avgs])
    elif double: return np.concatenate([mid,pd+mid]),np.concatenate([avgs,avgs])
    else: return mid,avgs

#can consider def denseyear as separate function

def phasestep(iband,pd,pbins,det=False,med=False,double=True,color='black',err=True,retall=False,epoch=0,sterr=True,divide=False,label=''):
    '''Step function for phase-folded data
    To do: ability to plot on input set of axes rather than creating new plot'''
    #use mean phase to get middle values of bins and means in each bin
    mid,avg,std = meanphase(iband,pd,pbins=pbins,det=det,med=med,double=double,stdev=True,sterr=sterr,epoch=epoch,divide=divide)
    plt.step(mid,avg,where='mid',color=color,label=label)
    #add errors as one sigma
    if err: plt.errorbar(mid,avg,yerr=std,color=color,marker='',linestyle='none',alpha=0.4)
    #flip y axis 
    maxa,mina = np.nanmax(avg),np.nanmin(avg)
    if err:
        maxa += std[np.argmax(avg)]
        mina -= std[np.argmin(avg)]
    plt.ylim(maxa+.01,mina-.01)
    plt.ylabel('I mag')
    plt.xlabel('Phase (days)')
    if retall and err: return mid,avg,std
    elif retall: return mid,avg
    else: return
    
    #return inds of densest year
def finddense(tab,maxspace=20,retsample=False,retall=False):
    '''Find largest region with dense sampling'''
    #spacing between points
    samp = tab['MJD-50000'][1:]-tab['MJD-50000'][:-1]
    #indices of sampling where spacking is greater than maxspace (default 20 days)
    sinds = np.where(samp>maxspace)[0]
    #add 1 to each value to make indexing work
    sinds += 1
    #number of points between gaps of maxspace
    sdiff = sinds[1:]-sinds[:-1]
    #starting index of densest/longest part (most points without a break of maxspace or more)
    stind = np.argmax(sdiff)
    if retall:
        #return all dense regions and index of most dense
        return sinds,stind
    #find average sampling to inform periodogram
    elif retsample:
        st = sinds[stind]
        end = sinds[stind+1]
        #mean sampling in this region
        msample = np.mean(tab['MJD-50000'][st+1:end]-tab['MJD-50000'][st:end-1])
        return (st,end),msample
    else: return sinds[stind:stind+2]

def findpeaks(freq,power,retsorted=False,sigma=0,height=0.05,distance=30,div=2,pkorder=False):
    #put period and power of periodogram results into DataFrame
    df = pd.DataFrame(columns=['period','power'])
    df['period'] = 1/freq
    df['power'] = power
    #automatically set threshold height using 3 sigma
    if sigma>0:
        #want to find med and stdev of power (only including low powers)
        #length of df
        lenf = len(df)
        df_pow = df.sort_values(by='power',ascending=False)
        #safe as not including peaks
        #can change div to include more or less in median and stdev determination
        low = df_pow[int(lenf/div):]
        #stdev
        std = np.std(low['power'])
        #find median value of power
        medpow = np.median(low['power'])
        #set height to sigma over median
        height = medpow+sigma*std
    #identify peaks
    peaks = signal.find_peaks(df['power'],height=height,distance=distance) 
    
    #sort peaks by power and return
    if pkorder:
        #data frame of peaks
        pf = pd.DataFrame(columns=['period','power','ind'])
        pers = []
        pers = np.array(df.loc[peaks[0]]['period'])
        pf['period'] = pers
        pf['ind'] = peaks[0]
        pf['power'] = peaks[1]['peak_heights']
        pf = pf.sort_values(by='power',ascending=False)
        pf.index = np.arange(len(pf))
        return pf
    #return DataFrame sorted by power and peaks dictionary
    elif retsorted: return df_pow,peaks
    #default: just return peaks dictionary
    else: return df,peaks #either need df or use inds of peaks before returning
    
def aliasarr(arr,nrange=1,cutzero=True):
    '''Find aliases above 1 (or given value mina) using array of periods
    arr: array of period (peaks)
    nrange: maximum n to use in alias calculation
            n's used: -1*nrange to nrange'''
    n = np.arange(-1*nrange,nrange+1) #includes zero so returned array will include periods passed in
    #repeat range of n for # of periods passed in
    nt = np.tile(n,(len(arr),1))
    nt = np.swapaxes(nt,0,1)
    a = np.tile(arr,(len(n),1))
    farr = nt + 1/a 
    parr = np.abs(1/farr)
    #cut out array of original periods (from n=0)
    if cutzero:
        parr = np.concatenate([parr[:nrange],parr[nrange+1:]])
    return parr

def findbeat(p1,p2):
    '''Return beat period given two periods'''
    return 1/np.abs(1/p1 - 1/p2)

def multiphase(tab,st=0,end=-1,dense=True,orb=10,incl_orb=True,meanp=True,sigma=20,distance=30,minp=5,maxp=100,
               pbins=10,maxspace=20,plotpd=False,color='darkseagreen',top5=True,pkorder=False,samples=10):
    '''Uses findpeaks to run periodogram, find peaks, and then phase folds with each one as well as, 
    optionally, the known orbital period.
    
    dense: use finddense to find dense LC region 
    meanp: include mean phase on plots
    pbins: number of phase bins for meanphase
    top3: only takes top three peaks
    
    TO DO: add detrend option
    '''
    if dense:
        dinds = finddense(tab,maxspace=maxspace)
        st,end = dinds[0],dinds[1]
    print(f'start ind: {st}, end ind: {end}')
    #runs periodogram
    freq,power,bp = periodogram(tab[st:end],more=True,minp=minp,maxp=maxp,plot=plotpd,samples=samples)
    #runs findpeaks, which finds periodogram peaks and returns df of period and power and peaks tuple
    df,pks = findpeaks(freq,power,sigma=sigma,distance=distance)
    numpk = len(pks[0])
    if top5 and numpk>5:
        #only keep top five peaks
        pks
        numpk = 5
        pf = pd.DataFrame(columns=['ind','pow'])
        pf['ind'],pf['pow'] = pks[0],pks[1]['peak_heights']
        pf = pf.sort_values(by='pow',ascending=False)
        pf = pf[:5] #only keep top 5 power values
        pinds = np.array(pf['ind'])
        pows = np.array(pf['pow'])
    else:
        pinds = pks[0]
        pows = pks[1]['peak_heights']
    if incl_orb:
        numpk+=1   
    if numpk<6:fig,ax = plt.subplots(1,numpk,figsize=(numpk*4,3),sharey=True)
    elif top5:fig,ax = plt.subplots(1,6,figsize=(6*4,3),sharey=True) 
    else: fig,ax = plt.subplots(1,numpk,figsize=(numpk*4,3),sharey=True)
    plt.subplots_adjust(wspace=0.05)
    if numpk == 1: return df,pks
    for i in range(numpk):
        if i == 0 and incl_orb:
            p = orb
            label = 'orb pd: '+str(p)
        else:
            pind = pinds[i-1]
            p = float(df['period'][pind:pind+1])
            po = pows[i-1]
            label = f'{p:.2f} pow: {po:.2f}'
        #in this case was just using densest region
        ax[i].scatter(tab['MJD-50000'][st:end]%p,tab['I mag'][st:end],color=color,s=4,label=label)
        ax[i].scatter(p+tab['MJD-50000'][st:end]%p,tab['I mag'][st:end],color=color,s=4)
        if meanp:
            mid,avgs = meanphase(tab[st:end],p,pbins=pbins,det=False)
            ax[i].plot(mid,avgs,color='black')
            ax[i].plot(p+mid,avgs,color='black')
        ax[i].legend()
    ax[0].set_ylim(np.max(tab['I mag'][st:end])+.01,np.min(tab['I mag'][st:end])-.01)
    if pkorder:
        #data frame of peaks
        pf = pd.DataFrame(columns=['period','power'])
        pers = []
        for p in pks[0]: #indexing only working with loop
            pers.append(float(df[p:p+1]['period']))
        pf['period'] = pers
        pf['power'] = pks[1]['peak_heights']
        pf = pf.sort_values(by='power',ascending=False)
        return pf
    
    else: return df,pks

def denselcpd(tab,dense,orb=0,minp=5,maxp=100,figsize=(22,14),minpoints=30,color='palevioletred',plotbest=False,onlybp=False,det=False,window=31):
    '''Use indices of dense regions (finddense) to plot subplots with LC chunks and inset periodograms
    dense: array of inds of dense regions from finddense, or other array of indices to use
    maxp: maximum period in periodogram search
    minpoints: minimum number of points in a region
    
    returns period and power of top two peaks from each periodogram'''
    fig = plt.figure(figsize=figsize)
    p = 1 #separate counter to not leave blank spaces
    bps = []
    sbp = [] #second best period
    maxpows = []
    spows = []
    stdate = []
    endate = []
    ndense = len(dense)
    rows = ndense/4
    for d in range(1,len(dense)):
        if dense[d]-dense[d-1]>minpoints:
            st = dense[d-1]
            end = dense[d]
            ttab = tab[st:end]
            #detrend data 
            if det:
                detrend(ttab,window=window,plot=False)
                imag = ttab['I detrend']
            else: imag = ttab['I mag']
            ax = fig.add_subplot(rows,4,p)
            ax.scatter(ttab['MJD-50000'],imag,color=color,s=10)
            maxi,mini = np.max(imag),np.min(imag)
            ax.set_ylim(maxi+.04,mini-.04)
            #instet periodogram
            axins = inset_axes(ax, width=figsize[0]/20, height=figsize[1]/20,borderpad=0.5) 
            freq,power,bp = periodogram(ttab,minp=minp,maxp=maxp,det=det,more=True,plot=False)
            axins.plot(1/freq,power,color='black')
            p+=1
            bps.append(float(bp))
            maxpows.append(np.max(power))
            stdate.append(float(tab['MJD-50000'][st:st+1]))
            endate.append(float(tab['MJD-50000'][end-1:end]))
            #find second best period (returns df of period and power with descending power)
            pf = findpeaks(freq,power,sigma=10,distance=10,pkorder=True)
            if not onlybp:
                sp = float(pf['period'][1:2])
                spow = float(pf['power'][1:2])
                sbp.append(sp)#append second highest
                spows.append(spow)
    stdate = np.array(stdate)
    endate = np.array(endate)
    if plotbest:
        fig = plt.figure(figsize=(5,3))
        plt.scatter(stdate+(endate-stdate)/2,bps,c=maxpows)
        plt.errorbar(stdate+(endate-stdate)/2,bps,xerr=(endate-stdate)/2,c='grey',alpha=0.4,linestyle='none')
#         plt.axhline(orb,color='grey')
        plt.colorbar(label='Power')
        plt.ylabel('Best Period')
        plt.xlabel('MJD-50000')
        if orb>0: plt.axhline(orb,alpha=0.4)
    if onlybp: return bps,maxpows,np.array(stdate),np.array(endate)
    else: return bps,maxpows,sbp,spows,stdate,endate

def yrpd(iband,minp=5,maxp=100,orb=0,plotbest=True,det=False,window=81,plotpd=False,spline=False,btol=50,sects=365):
    '''One periodogram per year
    returns years (indices of year bounds) and list best periods'''
    #make tab for each year in LC
    years = []
    stdate = iband['MJD-50000'][0]
    endate = iband['MJD-50000'][-1]
    y = 1
    while y < int((endate-stdate)/sects)+1:
        #less than next year
        year = iband[iband['MJD-50000']<stdate+sects*y]
        #also more than previous
        year = year[year['MJD-50000']>stdate+sects*(y-1)]

        years.append(year)
        y+=1
    #make it easy by assuming max possible years and fill in
    if plotpd: fig = plt.figure(figsize=(22,16))
    bps = []
    p = 1
    for y in years:
        if spline:
            splinedetrend(y,window=window,btol=btol)
        elif det:
            if len(y)>window: detrend(y,window=window)
        freq,power,bp = periodogram(y,minp=minp,maxp=maxp,more=True,plot=False,det=det)
        bps.append(float(bp))
        if plotpd:
            ax = fig.add_subplot(4,6,p)
            ax.plot(1/freq,power,color='black')
            if orb>0:ax.axvline(orb,color='darkseagreen',alpha=0.5)
        p+=1
    if plotbest:
        fig,ax = plt.subplots(1,1,figsize=(4,3))
        ax.scatter(np.arange(len(years)),bps,color='black')
        if orb>0:ax.axhline(orb,color='darkseagreen',alpha=0.5)
        ax.set_ylabel('Best Period (days)')
        ax.set_xlabel('Year Number')
        #residual plot
#         ax1.scatter(np.arange(len(years)),np.array(bps)-orb,color='black')
#         ax1.axhline(0,alpha=0.5)
#         ax1.set_ylabel('Residuals (Best - Est. Period)')
    return years,bps

def rollpd(iband,npoint=200,nroll=20,det=False,minp=20,maxp=120,plot=False,plotbest=True):
    '''Perform LS periodogram on a rolling basis, i.e. move indices of search by nroll,
    which is less than npoint (the number of points used in a search)'''
    bps = []
    ps = []
    pows = []
    st = 0
    sts = []
    maxps = []
    while st+npoint<len(iband):
        freq,power,bp = periodogram(iband[st:st+npoint],det=det,minp=minp,maxp=maxp,more=True,plot=plot)
        ps.append(1/freq)
        pows.append(power)
        maxps.append(np.max(power))
        bps.append(float(bp))
        sts.append(iband['MJD-50000'][st:st+1])
        st+=nroll
    if plotbest:
        plt.figure(figsize=(5,4))
        plt.scatter(sts,bps,c=maxps)
        plt.colorbar(label='LS Power')
        plt.xlabel('Start Date (MJD-50000)')
        plt.ylabel('Best Period')
    return ps,pows,bps,sts

def detrend(tab,window=201,printall=False,plot=False,figsize=(4,3),size=3):
    Imag = tab['I mag']
    if printall: print('Smooth (window = ', window, ') and detrend data...')
    Ismooth = signal.savgol_filter(Imag, window, 1)
    Imean = np.mean(Imag)
    if printall: print('Average I band magnitude', Imean)
    tab['I detrend'] = Imag-Ismooth  + Imag.mean()

    if printall: print('min:',i26['I detrend'].min(),'max:',i26['I detrend'].max())
    if plot:
        fig = plt.figure(figsize=figsize)
        plt.scatter(tab['MJD-50000'],tab['I mag'],color='black',label='original',s=size)
        plt.scatter(tab['MJD-50000'],tab['I detrend'],color='darkseagreen',label='detrended',s=size)
        plt.legend()
        
def splinedetrend(tab,window=201,btol=50,retspline=False):
    '''Add detrended I mag as I detrend in table
    retspline (bool): return flatten, trend from wotan.flatten'''
    flatten, trend = wotan.flatten(tab['MJD-50000'],tab['I mag'],method='rspline',window_length=window,break_tolerance=btol,return_trend=True)
    #change nans to zeros -- but figure out way to ignore so as to not add false values to periodogram
    trend[np.isnan(trend)] = 0
    #add detrended column: original - trend + mean
    mean = np.mean(tab['I mag'])
    detr = tab['I mag'] - trend + mean
    #replace outliers (more than a magnitude from mean) with mean
    detr[np.abs(detr-mean)>1] = mean
    tab['I detrend'] = detr
    if retspline:
        return flatten,trend
    
#spline detrending
def splinesearch(srcn,cross,full,minp=5,maxp=100,det=True,window=200,both=True,btol=50,phase=True,color='black',ylim=.06,close=False,mlist=['OII I','OIII I'],calib=False):
    '''Load in light curve and plot; spline detrend, and search for orbital period'''
    #get I and V LCs and plot
    plot = not close
    if both:iband,vband = getIV(srcn,cross,plot=plot,zooms=False,both=both,figsize=(8,4),mult=(3,8),offset=10,stack=True,save=False,mlist=mlist,calib=calib)
    else:iband = getIV(srcn,cross,plot=plot,zooms=False,both=both,figsize=(8,4),mult=(3,8),offset=10,stack=True,save=False,mlist=mlist,calib=calib)
    row = full[full['src_n']==srcn]
    #established period
    orb = float(row['Porb'])
    print(f'established period: {orb}')
    time = iband['MJD-50000']
    flux = iband['I mag']
    #detrend with rspline
    flatten_lc1,trend_lc1 = splinedetrend(iband,window=window,btol=btol,retspline=True)
    plt.plot(time, trend_lc1, color='black', linewidth=1, label='rspline')
    if close: plt.close()
    #periodogram with detrended
    if det: 
        bp = periodogram(iband,det=True,minp=minp,maxp=maxp)
        mag = 'I detrend'
    #periodogram without detrending:
    else: 
        bp = periodogram(iband,det=False,minp=minp,maxp=maxp)
        mag = 'I mag'
    plt.axvline(orb,color='darkseagreen')
    if close: plt.close()
    #phase-fold detrended I band with best period
    if phase and not close:
        plt.figure(figsize=(7,5))
        plt.scatter((iband['MJD-50000']%bp)/bp,iband[mag],color=color,alpha=0.5)
        plt.scatter(1+(iband['MJD-50000']%bp)/bp,iband[mag],color=color,alpha=0.5)
        medi = np.median(iband[mag])
        mid,avg,err = meanphase(iband,bp,pbins=16,det=det,double=True,stdev=True,epoch=0,divide=True,sterr=True)
        plt.step(mid,avg,color='black',where='mid')
        plt.errorbar(mid,avg,color='black',yerr=err,ls='none')
        plt.ylim(medi+ylim,medi-ylim)
        plt.ylabel('I mag',fontsize=14)
        plt.xlabel(f'Phase ({bp:.2f}d)',fontsize=14)
        if close: plt.close()
    if both: return iband,vband,bp
    else: return iband,bp
        
def detline(tab,st=0,end=-1,plot=False,figsize=(12,4),color='palevioletred',size=5,addmean=True):
    '''Detrend I mag with linear fit
    Returns I mag - linear fit
    addmean: bool to add mean I mag value after subtraction'''
    ttab = tab[st:end]
    mod = np.polyfit(ttab['MJD-50000'],ttab['I mag'],1)
    linmod = ttab['MJD-50000']*mod[0]+mod[1]
    if addmean: lindet = ttab['I mag'] - linmod + np.mean(ttab['I mag'])
    else: lindet = ttab['I mag'] - linmod
    if plot:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=figsize)
        #plot original data
        ax1.scatter(ttab['MJD-50000'],ttab['I mag'],color=color,s=size)
        #flip axes
        maxi,mini = np.max(ttab['I mag']),np.min(ttab['I mag'])
        ax1.set_ylim(maxi+0.02,mini-0.02)
        
        #plot detrended data
        ax2.scatter(ttab['MJD-50000'],lindet,s=size,color=color)
        #flip axes
        maxi,mini = np.max(lindet),np.min(lindet)
        ax2.set_ylim(maxi+0.02,mini-0.02)
    return lindet


def periodogram(tab,det=False,more=False,minp=5,maxp=30,bayes=False,sub=False,figsize=(4,3),plot=True,dodetrend=False,wfunc=False,window=11,samples=10,color='black'):
    '''Perform and plot single LS periodogram.
    Two different return options.
    wfunc: plot window function, so set all y to 1 in periodogram'''
    
    t = tab['MJD-50000']
    if dodetrend:
        #decide whether to actually modify tab or create copy just for periodogram
        detrend(tab,window=window)
    if wfunc: y = 1
    elif det: y = tab['I detrend']
    else: y = tab['I mag']
#     dy = tab['I mag err']
    minf = 1./maxp
    maxf = 1./minp
    ls = LombScargle(t, y)
    freq, power = ls.autopower(normalization='standard',
                           minimum_frequency=minf,
                           maximum_frequency=maxf,
                           samples_per_peak=samples)
    if bayes: power = np.exp(power)
        
    best_freq = freq[np.argmax(power)]

    if plot:
        fig = plt.figure(figsize=figsize)
        plt.plot(1/freq,power,color=color)
        plt.xlabel('Period',fontsize=14)
        plt.ylabel('Power',fontsize=14)
        #put text with best period
        plt.text(minp+(maxp-minp)/2,0.8*np.max(power),f'{1/best_freq:.2f}')
    if more:
        return freq, power, 1/best_freq
    else:
        return 1/best_freq
#-----------------------------------------------------------------FITTING PHASE-FOLD----------------------------------------------------------
def combine(srcn,cross,full,iband=[],pbins=16,det=True,pd=0,window=200,btol=50,minp=5,maxp=100,testbins=True,retstep=False,close=False,mlist=['OII I','OIII I']):
    #LC, detrend, and spline search
    if len(iband)==0: #if iband passed in, don't need to do splinesearch (but bp should then be >0 as well)
        iband,vband,bp = splinesearch(srcn,cross,full,close=close,minp=minp,maxp=maxp,det=det,window=window,btol=btol,phase=True,color='black',ylim=.08,mlist=mlist)
    #option to override bp with argument
    if pd > 0: bp = pd
    #analyze phase-folded data with best period, yielding dictionary
    plot = not close
    mid,avg,err,pdict = phase_dict(iband,bp,pbins,det=det,retstep=True,plotsymm=plot,close=close)
    if testbins: bindf = test_bins(iband,bp)
    if retstep: return mid,avg,err,pdict
    else: return pdict

def phase_dict(iband,pd,pbins,det=True,retstep=False,plotsymm=True,close=False):
    '''phase-fold all data and add amp, phase diff, phase max, phase min, shape, and diff sum to dictionary
    note: max and min are in mags, so phase max is phase bin # of faintest bin
    to do:  decide whether to convert to increasing from zero right away
    '''
    #initialize dictionary for properties of phase-folded data
    pdict = {}
    if close: mid,avg,err = meanphase(iband,pd,pbins,det=det,med=False,double=True,stdev=True,sterr=True,divide=True)
    else: mid,avg,err = phasestep(iband,pd,pbins,det=det,med=False,double=True,color='black',err=True,retall=True,epoch=0,divide=True,label='')
    #put in period for global analysis
    pdict['period'] = pd
    #calculate range
    pdict['amp'] = np.max(avg) - np.min(avg)
    #phase bin # of max and min bins
    maxp = np.where(avg == np.nanmax(avg))[0][0] 
    minp = np.where(avg == np.nanmin(avg))[0][0] 
    pdict['phase diff'] = (maxp-minp)/pbins
    if pdict['phase diff'] < 0:
        pdict['phase diff'] = 1+pdict['phase diff']
    pdict['phase max'] = mid[avg==np.nanmax(avg)][0]
    pdict['phase min'] = mid[avg==np.nanmin(avg)][0]
    #case 1 for FRED
    if pdict['phase diff']>0.5 and maxp>minp:
        pdict['shape'] = 'FRED'
    #case 2
    elif pdict['phase diff']<0.5 and maxp<minp:
        pdict['shape'] = 'FRED'
    else:
        #other shape: sine or symm?
        pdict['shape'] = 'not FRED'
    #redefine to use increasing numbers, start at 0 (decide whether to do this right away)
    diff = np.max(avg) - avg
    pdict['diff mean'] = np.mean(diff) #can compare mean to amp etc.; not dependent on I mag
    pdict['mean'] = np.mean(avg) #allows for luminosity correlations
    #whether diff or original avg used in skew just flips sign; kurtosis the same either way
#     print('remember skew flipped b/c of magnitudes')
#     print('so positive skew has asymmetry toward (more) low values = bright values')
    pdict['skew'] = st.skew(avg)
    pdict['kurtosis'] = st.kurtosis(avg)
    #adds symmetry sum to dictionary
    plt.figure(figsize=(5,4))
    summ = symm(mid,avg,err,pdict,pbins=pbins,plot=plotsymm)
    #add mean (standard) error
    pdict['mean err'] = np.mean(err)
    if retstep:
        return mid,avg,err,pdict
    else: return pdict
    
def symm(mid,avg,err,pdict,pbins=16,ylim=.01,square=False,plot=False):
    '''Fold over maximum; subtract and return sum of squares
    To do:  add propagation of standard error
            option to divide by some number (max in the bin? amplitude?)'''
    #roll data s.t. peak is at phase 0.5 --> makes fold easier
    roll = int(pbins/2) - int(float(pdict['phase min']*pbins-0.5))
    ravg = np.roll(avg,roll)
    rerr = np.roll(err,roll)
    sums = []
    st = pbins - 1
    if not plot: plt.close()
    for i in range(int(pbins/2)-1):
        sums.append(ravg[i]-ravg[st-i])
        if plot:
            plt.step(mid,ravg,color='black',where='mid')
            plt.errorbar(mid,ravg,yerr=rerr,ls='none',color='grey',alpha=0.5)
            maxa,mina = np.nanmax(avg),np.nanmin(avg)
            plt.ylim(maxa+.01,mina-.01)
            plt.scatter(mid[i],ravg[st-i],color='palevioletred')
            plt.scatter(mid[st-i],ravg[st-i],color='darkseagreen')
    sums = np.array(sums)
    if square: 
        pdict['symm sum'] = np.sum(sums**2)
        return np.sum(sums**2)
    else: 
        #'sum' but really an mean difference when folded in half around peak
        pdict['symm sum'] = np.abs(np.mean(sums))
        return np.sum(sums)
    
def test_bins(iband,bp,bins=np.arange(10,100,10),plot=True):
    bindf = pd.DataFrame(columns=['bins','amp','phase diff','phase max','phase min','shape',
                             'diff mean','mean','skew','kurtosis','symm sum','mean err'])
    bindf['bins'] = bins
    for b in bins:
        #temporary dictionary
        tdict = phase_dict(iband,bp,b,plotsymm=False)
        for d in list(tdict.keys()):
            row = bindf[bindf['bins']==b]
            row[d] = tdict[d]
            bindf[bindf['bins']==b] = row
    if plot:
        fig = plt.figure(figsize=(14,10))
        c = 1
        for r in ['phase diff','phase max','phase min','diff mean','mean','skew','kurtosis','symm sum','mean err']:
            ax = fig.add_subplot(3,3,c)
            ax.scatter(bindf['bins'],bindf[r],color='darkseagreen')
            ax.set_ylabel(r)
            if c==1:
                ax.set_xlabel('# phase bins')
            c+=1
        plt.subplots_adjust(wspace=0.3)
    #return df with row per #bins
    return bindf
#-----------------------------------------------------------------FLARE-FITTING--------------------------------------------------------------

    
def fit_sin(tt, yy,guess_freq=1/400.):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"
    Found this function online'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    #guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
#     guess_freq = 1/400.
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def addtotable(iband,vband,tab,num,interp,flag=False,det_std=0,det_iqr=0):
    '''Automatically update some columns of summary table
    flag: add 1 to calibration flag'''
    #row to update
    row = tab[tab['src_n']==num]
    #put in mean, stdev, max, min I and V band values
    row['mean I'] = np.mean(iband['I mag'])
    row['stdev I'] = np.std(iband['I mag'])
    row['min I'] = np.min(iband['I mag'])
    row['max I'] = np.max(iband['I mag'])
    row['I range'] = row['max I'] - row['min I']
    #repeat for V band
    row['mean V'] = np.mean(vband['V mag'])
    row['stdev V'] = np.std(vband['V mag'])
    row['min V'] = np.min(vband['V mag'])
    row['max V'] = np.max(vband['V mag'])
    row['V range'] = row['max V'] - row['min V']
    
    #correlation coefficients for Iint vs. V-I and V vs. V-I 
    icorr = np.corrcoef(vband['V mag']-interp,interp)[1][0]
    row['I V-I corr'] = icorr
    vcorr = np.corrcoef(vband['V mag']-interp,vband['V mag'])[1][0]
    row['V V-I corr'] = vcorr
    
    #redder when brighter(update criteria as needed)
    if icorr < -0.5 and vcorr < -0.5: row['redder when brighter'] = 'yes'
    elif icorr > 0.5 and vcorr > -0.5: row['redder when brighter'] = 'no'
    else: row['redder when brighter'] = 'check'
        
    #best fit lines to Iint vs. V-I and V vs. V-I 
    vi = vband['V mag']-interp
    imod = np.polyfit(vi,interp,1)
    #use coefficients to make string of equation
    #can decide if other format better and easier to use when pulling from table (e.g. breaking up into slope and int)
    row['I V-I slope'] = imod[0]
    row['I V-I int'] = imod[1] 
    #can add full = True to np.polyfit to also get sum of squares of residuals
    
    vmod = np.polyfit(vi,vband['V mag'],1)
    row['V V-I slope'] = vmod[0]
    row['V V-I int'] = vmod[1] 
    
    if flag: row['calibration flag'] = 1
    if det_std>0:
        row['det stdev I'] = det_std
        row['det stdev/I range'] = det_std/row['I range']
        #in this case, also update base num
        bn = (np.max(iband['I mag'])-np.median(iband['I mag']))-(np.median(iband['I mag'])-np.min(iband['I mag']))
        row['base num'] = bn
    if det_iqr>0:
        row['det I IQR'] = det_iqr
        

    #update table
    tab[tab['src_n']==num] = row
    
    #calculate I vs. V-I correlation to determine if redder when brighter
    #for now, redder when brighter just based on I vs. V-I, not both
    return 

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_increasing(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))

def monotonic(L,retwhich=False):
    inc,dec = False,False
    if non_increasing(L): dec = True
    if non_decreasing(L): inc = True
    if retwhich: return (inc or dec), inc #returns monotonic bool followed by bool for increasing
    else: return inc or dec
    
def cut(srcn,cross,cross2,mlist1,mlist2,cut=10,npoints=False,time=False,minp=1,retstd=True,retrange=False,calib=False,plot=True,text=False,statistic='median',glob=False,retsplit=False,window=200): #decide whether to do fixed chunks or fixed factor chunks or fixed number of points
    '''Divide LC into chunks and find median in each
    Division can be by number of points per piece or by total number of pieces.
    cut: number of pieces or number of points per piece or number of days per piece, depending on bool npoints and bool time
    npoints: if True, cut determines (approx) number of points per chunk; if False, cut determines total number of pieces
    time: if True, cut gives number of days included in each chunk
    retstd: return standard deviation of median values
    retrange: return range of median values
    calib: calibrate data by setting the medians of OII,OIII, and OIV to the same
    plot: plot LC
    retsplit: if True, return right after splitting; returns the split days,I mags, and det I lists
    minp: minimum points to check if time used (e.g. cut by year, but first check that there are minp points in each chunk)
    '''
    #make sure plt.text not used if not plotting
    if not plot: text = False
    if srcn in cross['src_n']:
        c = cross
        mlist = mlist1
    else:
        c = cross2
        mlist = mlist2
    if glob: 
        global iband
        iband = getIV(srcn,c,calib=calib,both=False,stack=True,plot=plot,mlist=mlist,printall=False)
    else: iband = getIV(srcn,c,calib=calib,both=False,stack=True,plot=plot,mlist=mlist,printall=False)
    #detrend with spline
    flatten,trend = splinedetrend(iband,retspline=True,window=window)
    if npoints: cut = int(len(iband)/cut)+1 #number of pieces is length of table divided by number of points in each piece
    #time cut means days per piece, so divide total time by that to get number of pieces
    if time: 
        imagsplit,splinesplit = [],[] #initialize lists of arrays
        if retsplit:
            timesplit,trendsplit = [],[]
        st_time = float(iband['MJD-50000'][:1])
        #loop through points and bin once time value hit
        c = 0
        min_used = 0 #counter to track how many chunks use min points rather than time
        prevind = 0
        for i in range(1,len(iband)):
            if (iband['MJD-50000'][i-1:i]>st_time+cut and c>minp) or i == len(iband):
                if c==minp+1: min_used+=1
                imagsplit.append(np.array(iband['I mag'][prevind:i]))
                splinesplit.append(np.array(iband['I detrend'][prevind:i])) 
                if retsplit:
                    timesplit.append(np.array(iband['MJD-50000'][prevind:i])) 
                    trendsplit.append(np.array(trend[prevind:i])) 
                
                c = 0 #reset counter for number of points in chunk
                prevind = i #save index as start of next chunk
                st_time = float(iband['MJD-50000'][i-1:i])#reset st_time 
            else: c+=1
        print(f'{min_used} chunks use min points rather than time')
    else:
        cut = int((iband['MJD-50000'][-1:]-iband['MJD-50000'][:1])/cut)
        imagsplit = np.array_split(iband['I mag'],cut)        
        splinesplit = np.array_split(iband['I detrend'],cut)
    #retsplit means just split up the values and return rather than calculating stat
    if retsplit:
        if not time:
            timesplit = np.array_split(iband['MJD-50000'],cut)
            trendsplit = np.array_split(trend,cut)
        return timesplit,imagsplit,splinesplit,trendsplit
    stats = []
    #TO DO: option to calculate statistic on detrended instead
    for i in range(cut):
        if statistic=='median': stat = np.median(imagsplit[i])
        elif statistic=='mean': stat = np.mean(imagsplit[i])
        elif statistic=='max': stat = np.max(imagsplit[i])
        elif statistic=='min': stat = np.min(imagsplit[i])
        elif statistic=='IQR': stat = scipy.stats.iqr(imagsplit[i])
        elif statistic=='stdev': stat = np.std(imagsplit[i])
        elif statistic=='skew': stat = scipy.stats.skew(imagsplit[i])
        elif statistic=='kurtosis':stat = scipy.stats.kurtosis(imagsplit[i])
        else: print('please enter valid statistic');return
        stats.append(stat)
    #add standard deviation of medians to plot
    if text:
        plt.text(np.median(iband['MJD-50000'])-1000,np.min(iband['I mag'])+.04,f'stdev of {statistic}s of {cut} chunks: {np.std(stats):.2f}')
    if retstd and retrange: return np.std(meds), np.max(meds)-np.min(stats)
    elif retstd: return np.std(stats)
    elif retrange: return p.max(stats)-np.min(stats)
    else: return stats
    
def gettype(tab,num='2'):
    '''Provided type column in tab, get list of source numbers with type
    equal to num
    num (str): type to get from tab
    tab: summ or allsumm
    returns typen (list): source numbers of that type'''
    typen = []
    for a in range(len(tab)):
        if tab.loc[a]['type']==num:
            typen.append(tab.loc[a]['src_n'])
    return typen

#BIG DIPS: SEPARATING TYPES 2 AND 3

def mono(smooth,minimum=4,one=False):
    num_mon = 0 #counter for monotonic
    i = 0 #begin with first point
    #number of immediate switches from monotonically increasing to monotonically decreasing
    #decide whether to also test the opposite
    nswitch = 0
    this_inc = False
    slocs = []
    onelocs = []
    #repeat for points after peak
    while i+3 < len(smooth): 
        sub = 2
        mono = True
        while mono and i+sub<len(smooth):
            #saves whether or not previous monotonic trend was increasing
            mono,inc = monotonic(smooth[i:i+sub],retwhich=True)
            #be careful since I band flipped
            if mono: this_inc = inc #final inc that's saved is during monotonicity
            sub+=1
        #add to num_mon based on what sub left off on; it automatically gets to 3
        if sub-1>minimum: #minimum 3 passes as long as sub is 4
            #subtracts 3 b/c sub automatically gets there
            num_mon += sub-3 #adds nothing if there aren't monotonic points of at least minimum
            #add to nswitch if this round was not increasing, last round was increasing, and last round is usable (min points) 
            if this_inc: #if it was increasing right before the start
                #look ahead to determine if it switches to monotonic decreasing
                #trying to only require 2
                mon,tinc = monotonic(smooth[i+sub-2:i+sub+(minimum-3)],retwhich=True)
                #put not tinc back in?
                if mon and not tinc: #if next three monotonically decreasing, add 1 to number of switches
                    nswitch+=1
                    slocs.append(i+sub-3)
            #other way of counting: minimum inc (fainter)
            if one and this_inc:
                onelocs.append(i+sub-3)
            elif one and not this_inc and i>3: #decreasing
                onelocs.append(i-1)
        #now reset i to i+sub to start again
        i += sub-2 #adds 1 if no monotonic behavior
    if one: return onelocs,num_mon
    else: return slocs,num_mon

#better to use scipy find_peaks rather than mono()

def bigdip(s,cross,cross2,mlist1,mlist2,ncut=30,npoints=False,time=False,minimum=4,statistic='median',plot=True,one=False,peaks=False,
           spline=False,sig=0,printall=False,printtype=True,frommin=False,checkbase=True,plotlc=False):
    '''
    spline (bool): use spline trend rather than cuts when searching for dips
    may need additional testing since moved from notebook
    '''
    if spline:
        try: iband, vband = getIV(t,cross,stack=True,plot=plot,mlist=mlist1,figsize=(4,3))
        except: iband, vband = getIV(t,cross2,stack=True,plot=plot,mlist=mlist2,figsize=(4,3))
        #set smooth variable to spline trend
        flatten, smooth = splinedetrend(iband,retspline=True)
    else:
        smooth = cut(s,cross,cross2,mlist1,mlist2,cut=ncut,npoints=npoints,time=time,retstd=False,retrange=False,calib=False,
                     plot=plotlc,text=False,statistic=statistic,glob=False,retsplit=False,window=200)
    #first check that faint (max) is more variable than bright (min)
    if checkbase:
        mincut = cut(s,cross,cross2,mlist1,mlist2,cut=10,npoints=npoints,time=time,retstd=False,retrange=False,calib=False,
                     plot=False,text=False,statistic='min',glob=False,retsplit=False,window=200)
        minstd = np.std(mincut)
        maxcut = cut(s,cross,cross2,mlist1,mlist2,cut=10,npoints=npoints,time=time,retstd=False,retrange=False,calib=False,plot=False,
                     text=False,statistic='max',glob=False,retsplit=False,window=200)
        maxstd = np.std(maxcut)
        if minstd > maxstd: #bright more variable
            t3 = False
            t2 = False
            if printtype: print('neither type 3 or type 2')
            return t3,t2      

    if plot:
        plt.figure(figsize=(4,3))
        plt.plot(np.arange(len(smooth)),smooth,color='black')
        maxi,mini = np.max(smooth),np.min(smooth)
        plt.ylim(maxi+.02,mini-.02)
    if peaks:
        pks = signal.find_peaks(smooth,height=0.2)
        slocs = pks[0]
        #also check in boundaries are dips (first and last point): condition is just being lower than neighbor
        if smooth[0]>smooth[1]:
            slocs = list(slocs)
            slocs.insert(0,0) #put 0 at beginning of location list
        #check end point
        if smooth[-1]>smooth[-2]:
            slocs = list(slocs)
            slocs.append(len(smooth)-1)
        slocs = np.array(slocs)
        if printall: print(slocs)
    else: slocs,num_mon = mono(smooth,minimum=minimum,one=one)
    #plot points of dips
    smooth = np.array(smooth) #allows for indexing in plot
    if plot: 
        plt.scatter(slocs,smooth[slocs],color='red')
    #sig higher than 0 indicates dips should be checked for outliers
    if sig>0:
        dips = smooth[slocs] #could probably do this check without peaks, but those can be useful anyway
        #upfilt corresponds to points fainter than sig sigma from median
        if frommin: start = np.min(smooth)
        else: start = np.median(smooth)
        upfilt = dips>start+sig*np.std(smooth)
        if plot: plt.axhline(start+sig*np.std(smooth),color='navy',label='dip cutoff',alpha=0.3)
        upout = dips[upfilt]
        uploc = list(np.where(upfilt)[0]) #+1 after [0] if indexing from 1 preferred
        #locations of sig sigma dips within all of smooth
        siglocs = slocs[upfilt]
        if printall: print(f'{len(upout)} dips {sig} sigma from median; locations {uploc}')
        if len(upout)>0: 
            t3 = True #possible Type 3; then will add in type4 bool below
            t2 = False #initialize Type 4 bool
        else:
            t3 = False
            t2 = False
        blist = [] #list of number of ~base values between dips
        #now, if there are > 1 dip, verify that smooth returned near bright base in between
        if len(upout)>1:
            #for each pair of dips
            for u in range(1,len(siglocs)):
                #smooth values between dips
                btwn = smooth[siglocs[u-1]:siglocs[u]]
                #filter to values that are close to overall min (bright)
#                 bright = btwn[btwn<np.median(smooth)+np.std(smooth)/4]
                #instead, just requires it goes above dip cutoff
                bright = btwn[btwn<start+sig*np.std(smooth)]
                if len(bright)>0: 
                    if printall: print(f'returns to bright base with {len(bright)} (of {ncut} total) ~base points between dips')
                    #also check that they're neighboring dips (can change condition to allow for 1 between?)
                    if uploc[u] - uploc[u-1] == 1: 
                        t3 = False
                        t2 = True
                else: 
                    if printall: print('does not return to base between dips so likely part of same dip, but checking for comparable depth')
                    #still can be real if the two dips are comparably faint
                    if np.abs(smooth[siglocs[u-1]]-smooth[siglocs[u]])<0.1:
                        if printall: print('still believable since close in mag (<0.1 mag diff)')
                        #also check that they're neighboring dips (can change condition to allow for 1 between?)
                        if uploc[u] - uploc[u-1] == 1: 
                            t3 = False
                            t2 = True
                blist.append(len(bright))
            #if plot: plt.axhline(np.median(smooth)+np.std(smooth)/4,color='darkseagreen',alpha=0.2,label='return cutoff')  
            if printall: print(blist)
        if plot: plt.legend()
            
        #minimum diff in variability
        #or cut out outliers and then look at std
#         if checkbase:
#             #sigma clip maxstd and minstd:
#             maxc_med,minc_med = np.median(maxcut),np.median(mincut) #medians of each cut
#             mincut,maxcut = np.array(mincut),np.array(maxcut)
#             #elements of maxcut that are within a sigma from median
#             smax = maxcut[maxcut>(maxc_med-1*maxstd)] #median - stdev
#             #elements of mincut that are within a sigma from median
#             smin = mincut[mincut<(minc_med+1*minstd)] #median + stdev
#             #now take standard deviations again -- should cut down to info about base itself
#             smaxstd = np.std(smax)
#             sminstd = np.std(smin)
#             print(f'new faint std: {smaxstd}, new bright std: {sminstd}')
#             ostd = np.std(smooth)
#             stdrat = sminstd/ostd
#             print(f'bright std/overall std:{stdrat}')
#             print(f'{smaxstd/sminstd} \n')
            
            
#         if checkbase:
#             if t2:
#                 if maxstd-minstd<0.1: 
#                     if printtype: print('neither type 3 or type 2')
#                     return False,False  
#             elif t3: #lower condition since more dips in t3
#                 if maxstd-minstd<0.08: 
#                     if printtype: print('neither type 3 or type 2')
#                     return False,False  
        if printtype: 
            if t3: print('Type 3')
            elif t2: print('Type 2')
            else: print('neither type 3 or type 2')
            
    if sig>0:return t3,t2

#PLOTTING TYPES
    
def tplot(typen,tab,text=False,label='1',marker='o',color='black',x='stdev I',y='det stdev I'):
    for t in typen:
        row = tab[tab['src_n']==t]
        if t==typen[0]:plt.scatter(row[x],row[y],marker=marker,label=label,color=color)
        #otherwise no label
        plt.scatter(row[x],row[y],marker=marker,color=color)
        if text: plt.text(row[x],row[y],str(t))