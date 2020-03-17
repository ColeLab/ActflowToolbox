def addNetColors_Seaborn(fcMatrix):
    
    """ A function to generate a Seaborn heatmap figure with CAB-NP colors added along axes of FC matrix; python 3
        INPUT 
        fcMatrix: a node x node matrix of FC estimates (in the Glasser parcellation, this would be 360 x 360, and presumably the 'grand mean' across subjects and states)
        Note: fcMatrix nodes should be sorted into their network order
        
        OUTPUT
        fig: a handle for the generated figure, can be used to save it, ex python code: 
            fig = addNetColors(fcMatrix)
            figDirectory = '/path/to/your/figure/directory/here/'; 
            figFileName = figDirectory + 'figureName.png'; fig.savefig(figFileName, bbox_inches='tight', format='png', dpi=250);
    """ 
    
    import numpy as np; 
    import matplotlib.pyplot as plt; 
    import matplotlib.patches as patches; 
    import seaborn as sns; 
    
    # CAB-NP & Glasser parcellation variables 
    colorList = [(0, 0, 1),(0.3922, 0, 1),(0, 1, 1),(0.6, 0, 0.6),(0, 1, 0),(0, 0.6, 0.6),(1, 1, 0),(0.98, 0.24, 0.98),(1, 0, 0),(0.7, 0.35, 0.16),(1, 0.6, 0),(0.25, 0.5, 0)];
    netBoundaries = [(0,5,6),(6,59,54),(60,98,39),(99,154,56),(155,177,23),(178,200,23),(201,250,50),(251,265,15),(266,342,77),(343,349,7),(350,353,4),(354,359,6)]; 
    [nParcels,nParcels] = np.shape(fcMatrix); 
    [numNets,c] = np.shape(colorList); 
    
    
    # Make room in FC matrix for network colors 
    bottomSize = (10,nParcels); topSize = (nParcels+10,10); 
    bottomBuff = np.zeros(bottomSize); topBuff = np.zeros(topSize); 
    bottomBuff = (bottomBuff+1)*0.31; topBuff = (topBuff+1)*0.31; # 0.31 is somewhat arbitrary, if it looks funny, change this number
    bottomAdd = np.vstack((fcMatrix,bottomBuff)); fcMatrixWithBuffer = np.hstack((bottomAdd,topBuff)); 
    np.fill_diagonal(fcMatrixWithBuffer, 0);
    
    #return fcMatrixWithBuffer; 

    # Generate figure 
    sns.set(style="white");
    fig, ax = plt.subplots(figsize=(10,7)); cmap = sns.diverging_palette(220, 10, as_cmap=True);
    tickLabels = np.arange(0,nParcels,50); 
    sns.heatmap(fcMatrixWithBuffer,center=0,cmap='seismic',square=True,cbar_kws={'label': 'FC Estimates'},xticklabels=tickLabels,yticklabels=tickLabels);
    plt.ylabel('Regions',fontsize=20); plt.xlabel('Regions',fontsize=20);
    plt.subplots_adjust(left=None, bottom=None, right=1, top=1, wspace=1, hspace=1);
    
    # Add network colors to the "buffered" axes 
    netList = list(range(numNets));
    for net in netList: 
        thisNet = netBoundaries[net]; netSize = thisNet[2]; netStart = thisNet[0]; 
        rectH = patches.Rectangle((netStart-1,359),netSize,10,linewidth=1,edgecolor=colorList[net],facecolor=colorList[net]); 
        rectV = patches.Rectangle((359,netStart-1),10,netSize,linewidth=1,edgecolor=colorList[net],facecolor=colorList[net]); 
        ax.add_patch(rectH); ax.add_patch(rectV);
    
    rectWhite = patches.Rectangle((nParcels-1,nParcels-1),10,10,linewidth=1,edgecolor='white',facecolor='white'); ax.add_patch(rectWhite); 
        
    # set global params & show image 
    plt.box(0); ax.set_xticks(tickLabels); ax.set_yticks(tickLabels); 
    plt.rc('ytick',labelsize=10); plt.rc('xtick',labelsize=10); 
    ax.tick_params(axis=u'both', which=u'both',length=0); plt.box(0); 
    plt.show()
    return fig;