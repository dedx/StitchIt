#
#This is the code library for the StitchIt project
#
# Copyright 2015 StitchIt - https://github.com/dedx/StitchIt
# 
# Author: J.L. Klay
# Date: 17-July-2015
#

#Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.ndimage as ndi
from scipy.misc import imresize
import requests
from StringIO import StringIO
from scipy.cluster.vq import kmeans,vq
from sklearn.utils import shuffle
import random
import datetime
from matplotlib.backends.backend_pdf import PdfPages


#Test function
def testme(verbosity = 1):
    '''Quick test function for creating default pattern

        Inputs:  Verbosity level.  If verbosity > 0, user will see extra output
        Returns: Nothing
        Outputs: pdf file with pattern called 'Pattern.pdf'
    '''
    imgurl = "http://upload.wikimedia.org/wikipedia/commons/1/1c/CMS_Higgs-event.jpg" #image file to pattern
    pattern_name = "Higgs Boson Event" #Name for pattern
    aidacolor = "black" #cloth color
    aidasize = 14 #number of stitches per inch of aida cloth
    reduct = 25 #Reduce the image size to this percent of the original
    numcol = 24 #number of colors to reduce image to
    
    #Retrieve image file
    response = requests.get(imgurl)
    before = ndi.imread(StringIO(response.content))
    #Reduce the size of the image
    smaller = resize(before,reduct)
    if verbosity > 0:
        plot_before_after(before,smaller,"Resized")
    #Reduce the number of colors in the image
    colors, counts, after = reduce_colors(smaller, numcol)
    if verbosity > 0:
        plot_before_after(before,after,"Color reduced")
    #Find the best-matched floss colors for the colors in the reduced image
    summary = floss_color_counts(colors,counts,aidasize,verbosity)
    #Replace colors in the image with matched floss colors
    for (skeins, floss, name, oldcolor, matchedcolor) in summary:
        replace_color(after,oldcolor,matchedcolor,verbosity)
    if verbosity > 0:
        plot_before_after(before,after,"Replaced color")
    #Show the original (reduced) colors and their matched floss colors
    if verbosity > 1:
        view_colors(summary)
    #Create the final pattern
    print "Creating your pattern in a file called Pattern.pdf.  Enjoy!"
    make_pattern(before,after,aidasize,imgurl,pattern_name,aidacolor)
    

#resize image
def resize(image,scale=25):
    '''Change the size of the image according to the scale and convert to
       integer RGB mode.
        Inputs: image, scale value
        Returns: reduced size RGB image
    '''
    return imresize(image,scale,mode="RGB")

#Kmeans algorithm to reduce number of colors
def reduce_colors(image, k):
    '''Apply kmeans algorithm.
        Input:   image, number of clusters to use
        Returns: colors, 
                 counts per color, 
                 new image
    '''
    if k > 32:
        print "Setting colors to maximum allowed of 32"
        k = 32
    rows, cols, rgb = image.shape
    # reshape the image in a single row array of RGB pixels
    image_row = np.reshape(image,(rows * cols, 3))
    #HERE ADD CODE TO GET A GOOD GUESS OF COLORS AND PASS THAT AS
    #SECOND ARGUMENT TO kmeans
    #image_array_sample = shuffle(image_row, random_state=0)[:1000]
    #kguess = kmeans(image_array_sample, k)
    #colors,_ = kmeans(image_row, kguess)
    # perform the clustering
    colors,_ = kmeans(image_row, k)
    # vector quantization, assign to each pixel the index of the nearest centroid (i=1..k)
    qnt,_ = vq(image_row,colors)
    # reshape the qnt vector to the original image shape
    image_centers_id = np.reshape(qnt,(rows, cols))
    # assign the color value to each pixel
    newimage = colors[image_centers_id]
    #count number of pixels of each cluster color
    counts,bins = sp.histogram(qnt, len(colors))
    return colors, counts, newimage

#JLK: Look into providing an array of RGB values corresponding to 
#available floss and using those to determine the clustering

#count colors
def color_count(image):
    '''Considering a (w,h,3) image of (dtype=uint8),
       compute the number of unique colors
    
       Encoding (i,j,k) into single index N = i+R*j+R*C*k
       Decoding N into (i,j,k) = (N-k*R*C-j*R, (N-k*R*C)/R, N/(R*C))
       using integer division\n

        Inputs:  image
        Returns: NumPy array of unique colors,
                 number of pixels of each unique color in image
    '''
    #Need to convert image to uint32 before multiplication so numbers are not truncated
    F = np.uint32(image[...,0])*256*256 + np.uint32(image[...,1])*256 + np.uint32(image[...,2])
    unique, counts = np.unique(F, return_counts=True)
    colors = np.empty(shape=(len(unique),3),dtype=np.uint32)
    numcol = np.empty(len(unique),dtype=np.uint32)
    i = 0
    for col,num in zip(unique,counts):
        R = col/(256*256)
        G = (col-R*256*256)/256
        B = (col-R*256*256-G*256)
        colors[i] = (R,G,B)
        numcol[i] = num
        i+=1
    return colors, numcol

#Found an online RGB<-->DMC floss color conversion table to reference.
def load_floss_colors(example=False):
    '''Load the library of floss colors from local file

        Inputs:  Boolean flag to show example color
        Returns: NumPy array of floss # and RGB values for each color 
                 List of color description string and hex color id
    '''
    values = np.loadtxt('DMCtoRGB_JLK.txt', delimiter=' , ',dtype=int, usecols=[0,2,3,4])
    labels = np.loadtxt('DMCtoRGB_JLK.txt', delimiter=' , ',dtype=str, usecols=[1,5])
    if example:
        print "Example: ",values[27],labels[27]
    return values,labels

#Use distance in RGB space to determine closest color match
def match_color(rgb,method="Euclidean"):
    '''For a given r,g,b tuple, determine the closest DMC thread color

        Inputs:  RGB color tuple, match method (Either "Euclidean" or "Metric")
        Returns: NumPy array of floss # and RGB values,
                 List of color description and hex color id
    '''
    values,labels = load_floss_colors(False)
    #Use color metric from http://www.compuphase.com/cmetric.htm
    rmean = (rgb[0]-values[:,1]) / 2.
    r = rgb[0]-values[:,1]
    g = rgb[1]-values[:,2]
    b = rgb[2]-values[:,3]
    match = np.sqrt(((2+rmean/256.)*r*r) + 4*g*g + (((2 + (255-rmean)/256.)*b*b)))
    #return values[match.argmin()],labels[match.argmin()]
             
    #Compute distance to nearest color in RGB space
    rdiff2 = (rgb[0]-values[:,1])**2
    gdiff2 = (rgb[1]-values[:,2])**2
    bdiff2 = (rgb[2]-values[:,3])**2
    cdiff = np.sqrt(rdiff2+gdiff2+bdiff2)
    #print "Input rgb: ",rgb,"\tClosest match:",values[cdiff.argmin()],labels[cdiff.argmin()]

    if method == "Euclidean":
        return values[cdiff.argmin()],labels[cdiff.argmin()]
    else:
        return values[match.argmin()],labels[match.argmin()]

#Determine the amount of floss needed
def yardskeincount(counts,aidasize):
    '''Given an array of stitch counts and an aida cloth
       size, calculate the number of skeins and yards of
       floss for each.  Includes a 20% buffer on the total to
       account for gaps, inefficient stitching

        Inputs:  NumPy array of number of pixels of each color, aida cloth size
        Returns: NumPy array of number of yards per color,
                 NumPy array of number of standard DMC skeins per color
    '''
    #floss length per stitch for given count aida cloth
    boxsize = 25.4/aidasize #mm
    #diagonal of box
    boxdiag = np.sqrt(2*boxsize**2)
    threadperstitch = 2*boxdiag + 2*boxsize
    threadperstitch *= 1.2 #increase by 20% to account for extras and gaps between stitches
    
    threadpercolor = counts*threadperstitch/1000. #number of meters of floss needed in each color
    #floss comes in skeins of 8.7 yards of 6-stranded thread.  Typical 14-count patterns take 2 strands
    flosslength = 3*8.7*0.9144 #convert yards of 6-stranded thread to meters of 2-stranded thread
    #yards of 6-stranded thread
    yardspercolor = threadpercolor*0.9144
    skeinspercolor = threadpercolor/flosslength # need to round up
    
    return yardspercolor,skeinspercolor

#Print and return the matched floss counts
def floss_color_counts(colors,counts,aidasize=14,verbosity=1):
    '''Sort the colors and counts according to the number of stitches per color,
       match the colors in the image to available floss colors and determine the 
       total amount of floss for each color needed to complete the pattern.

        Inputs:  NumPy array of the image colors, NumPy array of the color counts, aida cloth size, verbosity
        Returns: List containing 
                    # of skeins, 
                    DMC floss #, 
                    DMC floss color description, 
                    original color RGB tuple, 
                    Matched floss color RGB tuple
    '''
    #Sort by color counts
    mycounts = counts.copy() #copy so we don't mess up the original arrays
    inds = mycounts.argsort()
    sortedcolors = colors[inds].copy() #copy so we don't mess up the original arrays
    mycounts.sort()
    
    #Determine amount of floss needed for each color
    yds,skeinspercolor = yardskeincount(mycounts,aidasize)
    
    #Create a list of skein color count, code, and name
    summary = []
    if verbosity > 0:
        print "Counts\tRGBColor\tFloss#\tFlossRGB\t#Skeins\tFlossName"
        print "====================================================================================="    
    for i in range(len(counts)-1,-1,-1):
        matches = match_color(sortedcolors[i])
        if verbosity > 0:
            print mycounts[i],"\t",sortedcolors[i],"\t",matches[0][0],"\t",matches[0][1:],"\t %.2f"%skeinspercolor[i],"\t",matches[1]
        summary.append((float("%.2f"%skeinspercolor[i]),matches[0][0],matches[1][0],(sortedcolors[i]),(matches[0][1:])))
    return summary

#replace color at x,y coordinates (in pixel space) with floss color RGB to check the image fidelity
def replace_color(image,color,match,verbosity=1):
    '''Replace original colors in the image with the matched floss colors
       Modifies the image in place

        Inputs:  Image to modify, original color, replacement color, verbosity level
        Returns: Nothing
        Outputs: Modifies input image file
    '''
    if verbosity > 0:
        print color,"\t--->\t",match
    #find all indices in image with original color
    indices = np.where(np.all(image == color, axis=-1))
    #replace original color at indices in image with matched color
    image[indices[0],indices[1]] = match
    
#Print some useful pattern info
def aida_size(pic,aida=14,verbosity=0):
    '''Given an Aida cloth count and an image of a given number of pixels,
       report the size of the resulting pattern in inches

        Inputs: image, aidasize
        Returns: horizontal size of pattern in inches,
                 vertical size of pattern in inches
    '''
    #Image shape is (row,col): rows reflect size in y-dim and cols reflect size in x-dim
    y,x,col = pic.shape
    if verbosity > 0:
        print "Pixel dimensions: (%d x %d)"%(x,y)
        print "Aida Cloth count: %d"%aida
        print "Pattern dimensions: (%.2f in x %.2f in)"%(x/float(aida),y/float(aida))
        print "Pattern colors: %d"%(color_count(pic)[1]).size
    return x/float(aida), y/float(aida)

#Plot the image before and after manipulation
def plot_before_after(before,after,text="Transformed"):
    '''Plot side-by-side the before and after of a given image after being
       transformed in some way

        Inputs: Original image, transformed image, string description of transformation method
        Returns: Nothing
        Outputs: Matplotlib plot of image before and after
    '''
    #How many colors in the image before/after transformation
    cb = color_count(before)[1].size
    ca = color_count(after)[1].size
    # Two subplots, the axes array is 1-d
    f, axarr = plt.subplots(1,2,figsize=(12, 6))
    axarr[0].imshow(before)
    axarr[0].set_title('Original image (%d colors)'%cb)
    axarr[1].set_title('%s image (%d colors)'%(text,ca))
    axarr[1].imshow(after);
    plt.show();

#Dictionary of colors and symbols
def symbol_dictionary(colors):
    '''Create a dictionary of symbols for each color.  Black is assigned the symbol '.'
       There are 34 other available symbols.  The symbols are shuffled so that different
       patterns can have different symbols for the same color.

        Inputs: NumPy array of RGB color tuples
        Returns: Dictionary with RGB tuples as keys and symbols as elements
    '''
    #Available symbols chosen to be easily distinguishable and plottable as points
    #on a Matplotlib plot.
    symb = ['A','C','E','H','K','L','N','O','R','S','T','U','V','W','X','Y','Z',
            '1','2','3','4','5','7','8','9','-','+','=','/','*','(',')','<','>']
    symbols = []
    collist = []
    #Shuffle the symbols
    random.shuffle(symb,random.random)
    for i in range(len(colors)):
        #Treat black as a special color
        if np.all(colors[i] == (0, 0, 0)):
            symbols.append('.')
            collist.append((0,0,0))
        else:
            symbols.append(symb[i])
            collist.append((colors[i][0],colors[i][1],colors[i][2]))
    return dict(zip(collist,symbols))

def color_dictionary(after,aidasize,pattern_name,aidacolor):
    '''Create the pdf page showing the color list with floss labels and
       pattern symbols.  Needs to return the symbol dictionary for use by the
       pattern creator since the symbols are shuffled when the dictionary is created.

        Inputs:  Transformed image, aida cloth size, pattern name, fabric color
        Returns: Color/symbol dictionary for creating rest of pattern
        Outputs: Matplotlib plot sized for 8.5"x11" paper with pattern color/symbol info
    '''
    sizex,sizey = aida_size(after,aidasize,verbosity=0)
    colors,counts=color_count(after)
    yards,skeins = yardskeincount(counts,aidasize)
    values,labels = load_floss_colors(False)
    
    plt.figure(27,figsize=(7.5,10.))
    ax = plt.axes([0.025, 0.025, 0.95, 0.95],frameon=False)
    ax.set_xlim(0,1) 
    ax.set_ylim(0,1)
    plt.text(0.1,0.9,"Pattern Name:",fontweight='bold')
    plt.text(0.28,0.9,"%s"%pattern_name)
    plt.text(0.1,0.87,"Stitches:",fontweight='bold')
    plt.text(0.28,0.87,"%dw x %dh"%(after.shape[1],after.shape[0]))
    plt.text(0.28,0.85,"(2 strands, full)")
    plt.text(0.55,0.9,"Finished Size:",fontweight='bold')
    plt.text(0.73,0.9,"%.1f\"w x %.1f\"h "%(sizex,sizey))
    plt.text(0.55,0.87,"Fabric:",fontweight='bold')
    plt.text(0.73,0.87,"%s-ct %s aida"%(aidasize,aidacolor))
    plt.text(0.1,0.78,"DMC")
    plt.text(0.25,0.78,"Symbol")
    plt.text(0.4,0.78,"Color")
    plt.text(0.52,0.78,"Skeins")
    plt.text(0.65,0.78,"Description")
    plt.plot((0.08,0.9),(0.775,0.775),"k")
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.text(0.25,0.02,"$\copyright$ 2015 StitchIt - https://github.com/dedx/StitchIt")
    yspace = 0.03
    i = 1
    colsymb = symbol_dictionary(colors)
    for col,yc,sc in zip(colors,yards,skeins):
        #Find the index where color appears in the floss list
        index = np.where(np.all(values[:,1:] == col, axis=-1))
        plt.text(0.1,0.78-i*yspace,"%d"%values[index[0],0][0],ha='left')
        if (np.all(col == (0,0,0))):
            plt.plot(0.28,0.785-i*yspace,marker='.',color='k')
        else:
            #plt.text(0.278,0.78-i*yspace+0.003, colsymb[(col[0],col[1],col[2])], fontname='STIXGeneral',size=12, va='center', ha='center', clip_on=True)
            plt.plot(0.278,0.78-i*yspace+0.003,marker="$\mathrm{\mathsf{%s}}$"%colsymb[(col[0],col[1],col[2])],markersize=8,color='k')
        #plt.plot(0.282,0.78-(i-0.1)*yspace+0.0025, 's',markersize=10,markerfacecolor='None')
        plt.plot(0.425,0.78-(i-0.1)*yspace+0.0025, 's',markersize=10,markerfacecolor=(col[0]/255.,col[1]/255.,col[2]/255.))
        plt.text(0.56,0.78-i*yspace,"%.0f"%np.ceil(sc),ha='right')
        plt.text(0.65,0.78-i*yspace,"%s"%labels[index[0],0][0])
        
        i += 1
        
    return colsymb

#Compare matched and original colors
def view_colors(summary):
    '''Test function for viewing a comparison of the original (reduced) colors and the matched
       floss colors.  If the user prefers to substitute a color, they can do so using the replace_color function

        Inputs:  Summary list from floss_color_counts which contains
                    # of skeins, 
                    DMC floss #, 
                    DMC floss color description, 
                    original color RGB tuple, 
                    Matched floss color RGB tuple
        Returns: Nothing
        Outputs: Matplotlib plot showing all DMC floss colors and the list of original and matched colors
    '''
    vals,labs = load_floss_colors(example=False)
    fig = plt.figure(1,figsize=(15,15))
    plt.axes()

    count = 0
    #Plot original and matched colors
    for (skeins, floss, name, oldcolor, matchedcolor) in summary:
        circle = plt.Circle((count, 26), radius=0.5, fc=(oldcolor[0]/255.,oldcolor[1]/255.,oldcolor[2]/255.));
        plt.gca().add_patch(circle)
        plt.plot([-2,27],[25.5,25.5],"k")
        circle2 = plt.Circle((count, 25), radius=0.5, fc=(matchedcolor[0]/255.,matchedcolor[1]/255.,matchedcolor[2]/255.));
        plt.gca().add_patch(circle2)
        plt.text(count,24,floss,va='center', ha='center',fontsize=10)
        count += 1

    plt.text(17,25.75,"Original Colors",fontsize=15)
    plt.text(17,24.75,"Matched Floss",fontsize=15)
    plt.text(-6,22,"DMC Flosses:",fontsize=15)
    
    #Plot available floss colors
    for i in range(len(vals)):
        row=i/20
        col=i%20
        circle = plt.Circle((col, row), radius=0.5, fc=(vals[i,1]/255.,vals[i,2]/255.,vals[i,3]/255.));
        plt.gca().add_patch(circle)
        plt.text(col,row,vals[i,0],va='center', ha='center',fontsize=10)
        plt.axis('scaled')
    plt.xlim(-2,22)
    plt.ylim(-2,27)
    plt.show()
    
#retrieve x,y coordinates (in pixel space) of all pixels of a given color
def locate_color(image,color):
    '''Given an image, find all of the pixels that have the given input color
    
        Inputs:  Image to search, color to find
        Returns: NumPy arrays of col,row indices where this color is located in this image
    '''
    indices = np.where(np.all(image == color, axis=-1))
    #print zip(indices[0], indices[1])
    return indices[1], indices[0] #row,col vs. x,y  

#Create pattern coverpage
def cover_page(before,after,aidasize,imgurl,pattern_name,aidacolor):
    '''Create a cover page for this pattern showing the before/after images and
       basic info about the pattern

        Inputs:  Original image, transformed image, aida cloth size, image URL, pattern name, fabric color
        Returns: Nothing
        Outputs: Matplotlib plot sized for 8.5"x11" paper with pattern information
    '''
    sizex,sizey = aida_size(after,aidasize,verbosity=0)
    cb = color_count(before)[1].size
    ca = color_count(after)[1].size
    # Two subplots, the axes array is 1-d
    plt.figure(0,figsize=(7.5,10.))
    ax1 = plt.subplot2grid((3,2), (0,0), colspan=2,frameon=False)
    ax2 = plt.subplot2grid((3,2), (1,0))
    ax3 = plt.subplot2grid((3,2), (1,1))
    ax4 = plt.subplot2grid((3,2), (2,0), colspan=2,frameon=False)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0.5,1)
    ax1.text(0.475,0.85,"Pattern Name:",fontweight='bold',fontsize=15,ha='right')
    ax1.text(0.525,0.85,"%s"%pattern_name,fontsize=15)
    ax1.text(0.475,0.75,"Stitches:",fontweight='bold',ha='right')
    ax1.text(0.525,0.75,"%dw x %dh (2 strands, full)"%(after.shape[1],after.shape[0]))
    ax1.text(0.475,0.7,"Finished Size:",fontweight='bold',ha='right')
    ax1.text(0.525,0.7,"%.1f\" w x %.1f\" h "%(sizex,sizey))
    ax1.text(0.475,0.65,"Fabric:",fontweight='bold',ha='right')
    ax1.text(0.525,0.65,"%s-ct %s aida"%(aidasize,aidacolor))
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    
    ax2.imshow(before);
    ax2.set_title('Original image (%d colors)'%cb);
    ax2.set_xlabel("%d x %d pixels"%(before.shape[1],before.shape[0]))
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    
    ax3.set_title('Transformed image (%d colors)'%ca);
    ax3.set_xlabel("%d x %d pixels"%(after.shape[1],after.shape[0]))
    ax3.imshow(after);
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_yticklabels([])
    ax3.set_yticks([])
    
    ax4.set_xlim(0,1)
    ax4.set_ylim(0,1)
    ax4.text(0.5,0.85,"Original image url:",fontweight='bold',fontsize=6,ha='center')
    ax4.text(0.5,0.8,"%s"%imgurl,fontsize=6,ha='center')
    
    ax4.text(0.5,0.5,"$\copyright$ 2015 StitchIt - https://github.com/dedx/StitchIt",fontsize=15,ha='center')
    
    ax4.set_xticklabels([])
    ax4.set_xticks([])
    ax4.set_yticklabels([])
    ax4.set_yticks([])
    
#The Big Kahuna
def make_pattern(before,after,aidasize,imgurl,pattern_name,aidacolor):
    '''The complete pattern creator.  Uses 8.5"x11" sheets of paper.
       Divides the transformed image into pieces that are 80 pixels wide
       in x and 100 pixels wide in y and prints the symbols for each color
       on these pages.  Numbers the pages of the pattern from left to right along
       each row of the pattern.  Includes Cover page, symbol/color dictionary page,
       and pattern pages with copyright information in one multi-page pdf file that can
       be printed.
       
        Inputs:  Original image, Transformed image, aida cloth size, pattern name, fabric color
        Returns: Nothing
        Outputs: File called "Pattern.pdf" with the complete pattern pages
    '''
    sizex,sizey = aida_size(after,aidasize,verbosity=0)
    wp,hp = 7.5,10. #printable paper width,height
    cmperinch = 2.54 #cm
    boxespercm = 4 # 0.25mm per side
    numpix_x = int(np.floor(wp*cmperinch*boxespercm))
    numpix_y = int(np.floor(hp*cmperinch*boxespercm))
    gridx = 80 #max px in x direction
    gridy = 100 #max px in y direction
    border = 10 #grid box border
    xmax = after.shape[1]
    ymax = after.shape[0]
    pgsx = int(np.ceil(xmax/float(gridx)))
    pgsy = int(np.ceil(ymax/float(gridy)))
    pxperpg_x = np.zeros([pgsy, pgsx],dtype=int)+80
    pxperpg_y = np.zeros([pgsy, pgsx],dtype=int)+100
    #leftmost pages have 70 px in x, next n have 80 each, rightmost have remainder
    #topmost pages have 90 px in x, next n have 100 each, bottommost have remainder
    pxperpg_x[:,0] -= 10; pxperpg_x[:,-1] = xmax-pxperpg_x[0,0:-1].sum()
    pxperpg_y[0,:] -= 10; pxperpg_y[-1,:] = ymax-pxperpg_y[0:-1,0].sum()  

    finalunique, finalcount = color_count(after)
    
    #http://matplotlib.org/examples/pylab_examples/multipage_pdf.html
    # Create the PdfPages object to which we will save the pages:
    # The with statement makes sure that the PdfPages object is closed properly at
    # the end of the block, even if an Exception occurs.
    with PdfPages('Pattern.pdf') as pdf:
        cover_page(before,after,aidasize,imgurl,pattern_name,aidacolor)
        pdf.savefig(papertype='letter',bbox_inches='tight')
        print "Adding Cover page"
        plt.close()
        page = 0
        colsymb = color_dictionary(after,aidasize,pattern_name,aidacolor)
        pdf.savefig(papertype='letter',bbox_inches='tight')
        print "Adding page %d"%page
        plt.close()
        for row in range(pgsy): #row=y
            for col in range(pgsx): #col=x
                page += 1
                pxmin = pxperpg_x[row,:col].sum()
                pxmax = pxperpg_x[row,:col].sum()+pxperpg_x[row,col]-1
                pymin = pxperpg_y[:row,col].sum()
                pymax = pxperpg_y[:row,col].sum()+pxperpg_y[row,col]-1

                rowmin = pymin; rowmax = pymax
                colmin = pxmin; colmax = pxmax
                pltxmin = pxmin; pltxmax = pxmax
                pltymin = pymax; pltymax = pymin
                if col == 0:
                    pltxmin -= border
                if row == 0: 
                    pltymax -= border
                if col == pgsx-1:
                    pltxmax = gridx*pgsx
                if row == pgsy-1:
                    pltymin = gridy*pgsy
    
                plt.figure(row,figsize=(wp,hp))
                ax = plt.axes([0.025, 0.025, 0.95, 0.95])
                ax.set_xlim(pltxmin,pltxmax) #Row,Col vs. x,y
                ax.set_ylim(pltymin,pltymax) #with padding around image
                ax.xaxis.set_major_locator(plt.MultipleLocator(10.))
                ax.xaxis.set_minor_locator(plt.MultipleLocator(1.))
                ax.yaxis.set_major_locator(plt.MultipleLocator(10.))
                ax.yaxis.set_minor_locator(plt.MultipleLocator(1.))
                ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
                ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
                ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
                ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])
                for color in colsymb:
                    if (np.all(color) == 0):
                        x,y = locate_color(after[rowmin:rowmax+1,colmin:colmax+1,...],color)
                        #Use small dot for black pixels
                        plt.plot(x+pxmin+0.5,y+pymin+0.5, color='k',marker='.', lw = 0,markersize=2)
                    else:
                        x,y = locate_color(after[rowmin:rowmax+1,colmin:colmax+1,...],color)
                        #plt.text(x+pxmin+0.5,y+pymin+0.5, colsymb[color], fontname='STIXGeneral',size=12, va='center', ha='center', clip_on=True)
                        plt.plot(x+pxmin+0.5,y+pymin+0.5, marker="$\mathrm{\mathsf{%s}}$" % colsymb[color], markersize=4,color='k',lw = 0,markeredgecolor=None)
    
                ax.text(pltxmin+40,pltymin+4,"$\copyright$ 2015 StitchIt - https://github.com/dedx/StitchIt")
                ax.text(pltxmin,pltymin+4,"(%d,%d) pg %d of %d"%(row,col,page,pgsx*pgsy))
                pdf.savefig(papertype='letter',bbox_inches='tight')
                print "Adding page %d"%page
                plt.close()
                
                # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = 'StitchIt Pattern: %s'%pattern_name
                d['Author'] = 'J.L. Klay'
                d['Subject'] = 'Custom cross-stitch pattern file'
                d['Keywords'] = 'PdfPages cross-stitch multipage keywords author title subject'
                d['CreationDate'] = datetime.datetime(2015, 7, 17)
                d['ModDate'] = datetime.datetime.today()   
                
    print "All DONE!"


#A few extra utilities
def pagination(after):
    '''Show the pagination of the pattern for this image

        Inputs: Transformed image to create pattern from
        Returns: Nothing
        Outputs: Matplotlib plot of the paginated image
    '''
    sizex,sizey = aida_size(after) #use defaults?
    #How many pieces of paper do we need?
    gridx = 80 #max px in x direction
    gridy = 100 #max px in y direction
    border = 10 #grid box border
    pgsx = int(np.ceil(after.shape[1]/float(gridx)))
    pgsy = int(np.ceil(after.shape[0]/float(gridy)))

    print "Portrait:"
    print "xdim: %d pixels / %d pxperpg = %d pages"%(after.shape[1],gridx,pgsx)
    print "ydim: %d pixels / %d pxperpg = %d pages"%(after.shape[0],gridy,pgsy)

    pxperpg_x = np.zeros([pgsy, pgsx],dtype=int)+80
    pxperpg_y = np.zeros([pgsy, pgsx],dtype=int)+100
    #leftmost pages have 70 px in x, next n have 80 each, rightmost have remainder
    #topmost pages have 90 px in x, next n have 100 each, bottommost have remainder
    pxperpg_x[:,0] -= 10; pxperpg_x[:,-1] = after.shape[1]-pxperpg_x[0,0:-1].sum()
    pxperpg_y[0,:] -= 10; pxperpg_y[-1,:] = after.shape[0]-pxperpg_y[0:-1,0].sum()
    print pxperpg_x,"\n",pxperpg_x[0].sum()
    print pxperpg_y,"\n",pxperpg_y[:,0].sum()

    plt.figure(27,figsize=(4*sizex/sizey,4*sizey/sizex))
    ax = plt.axes([0,0,1,1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_major_locator(plt.MultipleLocator(1/4.))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1/3.))
    ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='k')
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='k')

    page = 0
    for row in range(pgsy):
        for col in range(pgsx):
            x = col/float(pgsx)+0.01
            y = 0.95-row/float(pgsy)
            pxmin = pxperpg_x[row,:col].sum()
            pxmax = pxperpg_x[row,:col].sum()+pxperpg_x[row,col]-1
            pymin = pxperpg_y[:row,col].sum()
            pymax = pxperpg_y[:row,col].sum()+pxperpg_y[row,col]-1
            ax.text(x,y,"(%d,%d)"%(row+1,col+1))
            page += 1
            ax.text(x+0.01,y-0.25,"pg %d"%page,fontsize=20)
            ax.text(x,y-0.1,"(%d:%d,"%(pxmin,pxmax))
            ax.text(x+0.05,y-0.15,"%d:%d)"%(pymin,pymax))

    #Visualize the different pages of the pattern
    row = 1; col = 1
    ymax = after.shape[0]
    pxmin = pxperpg_x[row,:col].sum()
    pxmax = pxperpg_x[row,:col].sum()+pxperpg_x[row,col]-1
    pymin = pxperpg_y[:row,col].sum()
    pymax = pxperpg_y[:row,col].sum()+pxperpg_y[row,col]-1
    rowmin = pymin; rowmax = pymax
    colmin = pxmin; colmax = pxmax
    print pxmin,pxmax,pymin,pymax
    print "pxperpg_y[:row,col].sum()",pxperpg_y[row:,col].sum()
    print ymax-pxperpg_y[row,col]
    fig = figure(5,figsize=(5*sizex/sizey,5*sizey/sizex))

    x,y = locate_color(after,(0,0,0))
    plt.plot(x,ymax-y+1,'k.')

    x2,y2 = locate_color(after[rowmin:rowmax+1,colmin:colmax+1,...],(0,0,0))
    plt.plot(x2+pxmin,ymax-y2-pymin+1,'r.')

    plt.xlim(0,276)
    plt.ylim(0,254)
