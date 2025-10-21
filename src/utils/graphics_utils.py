import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from num2tex import num2tex
from datetime import timedelta

from . import pandas_utils  as pu
import config as co




def numformat(n,prec=2):
    """Formats a number in scientific notation with a given precision
    Returns a LaTeX string

    Args:
        n (real): number to format
        prec (int, optional): precision. Defaults to 2.

    Returns:
        string: LaTeX string
    """

    if (n==0):
        return '0'
    if (np.abs(n)>10**(1-prec)):
        n_latex = f'${{:.{prec}g}}$'.format(num2tex(n,precision=prec))
    else:
        n_latex = f'${{:.{prec}e}}$'.format(num2tex(n,precision=prec))

    tag = r'$\\times '

    if n_latex[:len(tag)] == tag:
        n_latex = '$' + n_latex[len(tag):]
    
    return n_latex


def numformatlist(l,prec=2):
    formatted = [numformat(x,prec) for x in l]

def get_bins(listsample,Nbins,quantity=None,binmin=None,binmax=None,logax=False):
# logax to implement!
    if logax:
        if quantity is not None:
            binmin = listsample[0][quantity].min()       
            binmax = listsample[0][quantity].max()       
            for sample in listsample:
                binmin = np.min([binmin,sample[quantity].min()])
                binmax = np.max([binmax,sample[quantity].max()])
        else:
            binmin = listsample[0].min()       
            binmax = listsample[0].max()       
            for sample in listsample:
                binmin = np.min([binmin,sample.min()])
                binmax = np.max([binmax,sample.max()])
        my_min=np.log10(binmin)
        my_max=np.log10(binmax)
        my_bins = np.logspace(my_min, my_max, Nbins)
               
        
 
    else:    
    
        if binmin is None:
            if quantity is not None:
                binmin = listsample[0][quantity].min()       
                for sample in listsample:
                    binmin = np.min([binmin,sample[quantity].min()])
            else:
                binmin = listsample[0].min()       
                for sample in listsample:
                    binmin = np.min([binmin,sample.min()])
    
    
        if binmax is None:
            if quantity is not None:
                binmax = listsample[0][quantity].max()
                for sample in listsample:
                    binmax = np.max([binmax,sample[quantity].max()])
            else:
                binmax = listsample[0].max()
                for sample in listsample:
                    binmax = np.max([binmax,sample.max()])
    
    
        my_bins = np.linspace(binmin, binmax, Nbins)
    
    return my_bins

def plot_loghist(x, bins,ax):
    logbins = np.geomspace(x.min(), x.max(), bins)
    ax.hist(x, bins=logbins)
    ax.set_xscale('log')
    
    

def multidist(graphs,quantity,loc_ax,loc_bins, bordersup = 0, legendsize = 10, 
              style = "modern",alpha=1,linewidth=1,logax=False,
              putlabel=True):
         
    for index,row in graphs.iterrows(): 
        
        if (style == "simple"):
            dic_style={#"histtype": "step", 
                       "linewidth": linewidth, 
                       "alpha": 1, 
                       "color": row['colour']
                       }
            if putlabel:
                label = row['label']
                
                sns.histplot( data=row['df'], 
                              bins=loc_bins, 
                              label=label,
                              ax=loc_ax,
                              stat = "density",
                              element="step", 
                              fill=False,
                              **dic_style
                            );
            else:
                sns.histplot( data=row['df'], 
                              bins=loc_bins, 
                              ax=loc_ax,
                              stat = "density",
                              element="step", 
                              fill=False,
                              **dic_style
                            );
                
        else:
            dic_style={"linewidth": linewidth, 
                       "alpha"    : alpha,
                       # "edgecolor" : row['colour'],
                       "color"    : row['colour'],
                       "hatch"    : row['hatch']
                       }
            sns.histplot( data=row['df'], 
                          bins=loc_bins, 
                          label=row['label'],
                          ax=loc_ax,
                          stat = "density",
                          **dic_style
                        );

        # sns.distplot(data=row['df'], 
        #              x=quantity,
        #              bins=loc_bins, 
        #              hist=True, 
        #              kde=False, 
        #              rug=False,  
        #              label=row['label'],
        #              ax=loc_ax,
        #              hist_kws=dic_style
        #             )

        # sns.displot( data=row['df'], 
        #              x=quantity,
        #              bins=loc_bins, 
        #              # hist=True, 
        #              kde=False, 
        #              rug=False,  
        #              label=row['label'],
        #              ax=loc_ax
        #              # hist_kws=dic_style
        #             )



        # loc_ax.hist(row['df']/row['df'].sum(), 
        #             bins=loc_bins, 
        #             # label=row['label'],
        #             **dic_style
        #             )
    if logax:
        loc_ax.set_xscale('log')
    ymin,ymax = loc_ax.get_ylim()
    loc_ax.set_ylim(ymin,ymax+(ymax-ymin)*bordersup)
    
    if putlabel:
        loc_ax.legend(fontsize=legendsize);



def add_Wilcoxon(df1,df2,loc_ax,fig,
                 signif = 5e-2, signifcol = 'black', nonsignifcol = 'black',
                 legendsize = 12, annotation_margin=0.02, rank=0, padding = 1, 
                 suffix="",xpad=0, forcex = None, logx=False, label=True, single=True,
                 signifstyle = 'normal', nonsignifstyle = 'normal',forcealign=None):
                # supported styles: 'normal', 'italic', 'oblique'
        
    xmargin = 0.02
        
    fig.canvas.draw() # Otherwise, won't work!

    statistic, p_value = ranksums(df1,df2)
       
    # col = signifcol if (p_value <= signif) else nonsignifcol
    # style = 'normal' if (p_value <= signif) else 'oblique'
    col = signifcol if (p_value <= signif) else nonsignifcol
    style = signifstyle if (p_value <= signif) else nonsignifstyle
    
    p_latex = numformat(p_value)
    
    xmin,xmax = loc_ax.get_xlim()
    ymin,ymax = loc_ax.get_ylim()

    if not label:
        (xlocmin,ylocmin,xlocmax,ylocmax) = (xmin,ymin,xmax,ymax)
    else:
        leg = loc_ax.get_legend()
        bbox = leg.get_window_extent()
        inv = loc_ax.transData.inverted()
    
        (xlocmin,ylocmin)=inv.transform((bbox.x0,bbox.y0))
        (xlocmax,ylocmax)=inv.transform((bbox.x1,bbox.y1))
                
    
    if (forcex=="min"):
        if logx:
            xmargin = 0
        my_x = xmin + (xmax-xmin)*xmargin
        align = 'left'
    elif (forcex=="max"):
        if logx:
            xmargin = 0
        my_x = xmax - (xmax-xmin)*xmargin
        align = 'right'
    else:
        if ((xmax-xlocmax)<(xlocmin-xmin)):  #Legend on the right
            # my_x = xlocmax
            # align = 'right'
            my_x = xlocmax
            if single:
                align='right'
            else:
                align = 'left'
        else: #Legend on the left
            my_x = xlocmin
            align = 'left'

    if (forcealign):
        align=forcealign
    my_x = my_x + xpad  # automatize xpad!
    
    if not label:
        yloc_margin=ylocmax-(ymax-ymin)*(annotation_margin+padding*rank)   # automatize padding!
    else:
        yloc_margin=ylocmin-(ymax-ymin)*(annotation_margin+padding*rank)   # automatize padding!
    
      
    loc_ax.text(my_x,yloc_margin,"$p"+suffix+"=$%s"%p_latex, 
                family = sns.axes_style()['font.family'][0], 
                size = legendsize,
                style=style,
                color = col,
                horizontalalignment = align,
                verticalalignment='top'
               )

def putarrows(listdf,listcolours, loc_ax,
              err_bars = False, alpha = 1, arrowsize = 0.14,
              headarrow = 2.5e-2,headratio = 4,style="arrow",lw=1, 
              ymax=1):
    
    xmin,xmax = loc_ax.get_xlim()
    ymin,ymax = loc_ax.get_ylim()
        
    
    length = (ymax-ymin)*arrowsize
    headwidth = (xmax-xmin)*headarrow

    graphs=pd.DataFrame(
        {'df'     : listdf,
         'colour' : listcolours
        } 
    )

    for index,row in graphs.iterrows(): 
        med = row['df'].median()
        std = row['df'].std()
        if (style == "dashed"):
            loc_ax.axvline(med, 
                color=row['colour'], 
                linestyle='dashed', 
                lw=lw, 
                dashes = (5, 3, 1, 3),
                # ymax=ymax
                )
        else:
            loc_ax.arrow(med,ymax,0,-length,length_includes_head=True,
                       head_width=headwidth, head_length=length/headratio,
                       color=row['colour'],
                        alpha=alpha);


        if err_bars:
            loc_ax.errorbar(med,ymax-length, xerr=std, ecolor=row['colour'])
            

def savefig(fig,figfolder,name):
    fig.savefig(figfolder+name+'.pdf', format='pdf', bbox_inches='tight')
    
    
    
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

        
    
def form(x,nb_decimal=2):
    nb_str = str(nb_decimal)
    return '.' + nb_str + 'f' if (np.abs(x)>=10**(-nb_decimal)) else '.' + nb_str + 'e' 

def tex_form(x,nb_decimal=2,nb_decimal_exp=1, exp_min=6):
    neg_x = (x<0)
    x=np.abs(x)
    
    if(x<10**-exp_min):
        return f'$<10^{{-{exp_min}}}$'
    
    nb_str = str(nb_decimal)
    nb_exp_str = str(nb_decimal_exp)
    
    if (np.abs(x)>=10**(-nb_decimal)):
        my_form = '.' + nb_str + 'f'
        formatted = f'{x:{my_form}}$'
    else:
        my_form = '{:.' + nb_exp_str + 'e}$'
        formatted = my_form.format(num2tex(x,precision=nb_decimal_exp))
    
    if neg_x:
        formatted = '- ' + formatted
    
    formatted = '$' + formatted
    
    return formatted



import urllib
from IPython.display import Image
from IPython.display import display
# import cv2

cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'
def getgal(RA,Dec,impix = 10,imsize = 6): #imsize in arcsec
    query_string = urllib.parse.urlencode(dict(ra=RA,
                                               dec=Dec,
                                               width=impix, 
                                               height=impix,
                                               scale=imsize/impix))
    return cutoutbaseurl + '?' + query_string

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return imageRGB

def galim(row,ax,impix = 10,imsize = 6,labelsize=12):#imsize in arcsec
    RA = row['ra']
    Dec = row['dec']
    objid = row['objid']
    url = getgal(row['ra'],row['dec'],impix,imsize)
    img = url_to_image(url)

    ax.imshow(img)
    # label = f"RA={RA:3.3f}, Dec={Dec:3.3f}"
    label = f"{objid}"
    ax.set_title(label, fontsize=labelsize)    
    ax.set_xticks([])
    ax.set_yticks([])

          
        
def mosaic(df,size=20,Ncols=6,labelsize=12,imsize = 6,impix = 10, figname = ''):#imsize in arcsec
    Ntot = df.shape[0]
    if (Ntot%Ncols==0):
        Nrows = Ntot//Ncols
    else:
        Nrows = Ntot//Ncols + 1
        
    fig,axes = plt.subplots(nrows=Nrows, ncols=Ncols, figsize=(size,size)) 

    for index, ax in enumerate(axes.flat):
        if (index<Ntot):
            row = df.iloc[index]
            galim(row,ax,impix,imsize,labelsize)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    if (figname):
        gu.savefig(fig,co.figfolder,figname)


def format_time(T): # formats a time in seconds to h min s (s rounded)
    T_format = str(timedelta(seconds=T)).split(':')
    formated = f'{T_format[0]} h {T_format[1]} min {float(T_format[2]):.0f} s'
    return formated
    

def Nrows(Ntot,Ncols):
    # Gives the number of rows for Ntot objects arranged in Ncols columns
    if (Ntot%Ncols == 0):
        Nrows = Ntot//Ncols
    else:
        Nrows = Ntot//Ncols + 1
        
    return(Nrows)

def RowCol(index,Ncols):
    # Gives the coordinates (row,col) of index Ntot objects arranged in Ncols columns 
    row = index//Ncols
    col = index%Ncols
    
    return(row,col)


# Define custom y-axis tick formatter in LaTeX.
def scientific_formatter(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(x)))
    mantissa = x / (10**exponent)
    if np.isclose(mantissa, 1.0, atol=1e-2):
        return r'$10^{%d}$' % exponent
    else:
        return r'$%d\times10^{%d}$' % (int(round(mantissa)), exponent)


def mult_hist(data, figname=None, nb_bins=20, figsizex=12, figsizey=8, labelsize=24, 
              legendsize=22, ticklength=10, tickwidth=1):

        fig, ax = plt.subplots(figsize = (figsizex,figsizey))

        flattened_values = pu.flatten_list(data['Values'])
        my_min, my_max = min(flattened_values), max(flattened_values)

        bins = np.linspace(my_min, my_max, nb_bins)

        for i, (values, label, color, alpha) in enumerate(zip(data['Values'], data['Labels'], 
                                                          data['Colors'], data['Alpha'])):
                ax.hist(values, bins=bins, color=color, alpha=alpha, edgecolor='black', label=label)  

        plt.xlabel('$V_\mathrm{max}$',fontsize=labelsize)
        plt.ylabel('Count',fontsize=labelsize)
        plt.grid(False)
        ax.tick_params(labelsize=labelsize, direction='inout',axis='both', length=ticklength, width=tickwidth)
        ax.legend(fontsize=legendsize)
        ax.ticklabel_format(style='plain')

        plt.show()

        if figname is not None:
                savefig(fig,co.FIGURES_PATH,figname)

                