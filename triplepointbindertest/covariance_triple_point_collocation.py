def covariance_triple_point_collocation(data1, data2, data3, flag):
    

# Basic Triple Point Collocation Analysis for any combination of datasets
# Input datasets must be 2D with dimensions 1xN, Mx1, or MXN 
# All datasets must have uniform dimensions 
# Flag: 1 = exclude large dataset differences abs(>5PSU) from analysis, 0=include all dataset differences regardless of size

# J. Anderson (janderson@esr.org) based on Stoffelen 1998 & Gruber et al 2016 Covariance Notation 

    # Import packages if needed - ja check if need to do in function as well? look at examples
    import numpy as np
    import numpy.ma as ma
    
    # Error check input data for uniform dimensions and valid flag value
    # Python automaticaly checks that the correct numbe of imputs were given
    num = ((data1.shape == data2.shape)+(data1.shape == data3.shape)+(data2.shape == data3.shape))
    
    if num != 3:
        raise Exception('Input datasets must have uniform dimensions')

    if np.isin(flag,[0,1]) == False:
        raise Exception('Flag input must be 1 or 0')
        
    # Prepare data for triple point collocation
    # If flag = 1, remove data where absolute value of differences > 5 PSU
    if flag == 1:
        
        #'Data1 - Data3'
        ds_d1_d3 = data1-data3; # Calculate the difference between Data1 & Data3
        tempLarge = np.where(np.absolute(ds_d1_d3)>5)
        data1[tempLarge]='NaN'
        data3[tempLarge]='NaN'
        del tempLarge, ds_d1_d3
        
        #'Data2 - Data3'
        ds_d2_d3 = data2-data3; # Calculate the difference between Data1 & Data3
        tempLarge = np.where(np.absolute(ds_d2_d3)>5)
        data2[tempLarge]='NaN'
        data3[tempLarge]='NaN'
        del tempLarge, ds_d2_d3

        #'Data1 - Data2'
        ds_d1_d2 = data1-data2; # Calculate the difference between Data1 & Data3
        tempLarge = np.where(np.absolute(ds_d1_d2)>5)
        data1[tempLarge]='NaN'
        data2[tempLarge]='NaN'
        del tempLarge, ds_d1_d2
    
    # Triple point collocation cannot be calculated if one dataset has a NaN. 
    # Make location of NaNs uniform for all datasets. 
    data2[np.argwhere(np.isnan(data1))]='NaN'
    data3[np.argwhere(np.isnan(data1))]='NaN'

    data1[np.argwhere(np.isnan(data2))]='NaN'
    data3[np.argwhere(np.isnan(data2))]='NaN'

    data1[np.argwhere(np.isnan(data3))]='NaN'
    data2[np.argwhere(np.isnan(data3))]='NaN'
    
    # Start triple point collocation
    # Calculate variance of each dataset
    var_d1 = np.nanvar(data1,ddof=1) # Variance computed on flattened variance
    var_d2 = np.nanvar(data2,ddof=1)
    var_d3 = np.nanvar(data3,ddof=1)

    # Calculate the covariance of Data1 and Data3
    covar_d1_d3 = np.diagonal(ma.cov(ma.array(data1, mask=np.isnan(data1)), ma.array(data3, mask = np.isnan(data3)),ddof=1),offset=1) # covariance of data1 and data3 is off diagonal element, masked to ignore nans

    # Calculate the covariance of Data2 and Data3
    covar_d2_d3 = np.diagonal(ma.cov(ma.array(data2, mask=np.isnan(data2)), ma.array(data3, mask = np.isnan(data3)),ddof=1),offset=1) # covariance of data2 and data3 is off diagonal element, masked to ignore nans

    # Calculate the covariance of Data1 and Data2
    covar_d1_d2 = np.diagonal(ma.cov(ma.array(data1, mask=np.isnan(data1)), ma.array(data2, mask = np.isnan(data2)),ddof=1),offset=1) # covariance of data1 and data2 is off diagonal element, masked to ignore nans
    
    # Calculate the unscaled error variances for each dataset
    # Covariance is symmetrical, so cov(d1,d2)==cov(d2,d1)
    errorvar_d1 = (var_d1-((covar_d1_d2*covar_d1_d3)/covar_d2_d3))
    errorvar_d2 = (var_d2-((covar_d1_d2*covar_d2_d3)/covar_d1_d3))
    errorvar_d3 = (var_d3-((covar_d1_d3*covar_d2_d3)/covar_d1_d2))
    
    # Determine rescaling parameters if desired
    # Rescale to d1
    #rescale_d2 = (covar_d1_d3/covar_d2_d3)
    #rescale_d3 = (covar_d1_d2/covar_d2_d3)

    #errorvar_d1 = errorvar_d1
    #errorvar_d2 = errorvar_d2*(rescale_d2**2)
    #errorvar_d3 = errorvar_d3*(rescale_d3**2)

    # Calculate RMSD from the unscaled error variances
    rmsd_d1 = np.sqrt(errorvar_d1);
    rmsd_d2 = np.sqrt(errorvar_d2);
    rmsd_d3 = np.sqrt(errorvar_d3);

    # NumPy sqrt only returns sqrts for positive values, replace masked, invalid numbers with nans
    rmsd_d1 = rmsd_d1.filled(np.nan)
    rmsd_d2 = rmsd_d2.filled(np.nan)
    rmsd_d3 = rmsd_d3.filled(np.nan)

    # It is possible to have negative unscaled variances which means imaginarrmsd. Only keep positive variances, real rmsd.
    # Replace imaginary numbers with NaN
    #rmsd_d1(imag(rmsd_d1)~=0) = NaN; 
    #rmsd_d2(imag(rmsd_d2)~=0) = NaN; 
    #rmsd_d3(imag(rmsd_d3)~=0) = NaN;

    #rmsd_d1 = real(rmsd_d1);
    #rmsd_d2 = real(rmsd_d2);
    #rmsd_d3 = real(rmsd_d3);
    
    return [rmsd_d1,rmsd_d2,rmsd_d3]