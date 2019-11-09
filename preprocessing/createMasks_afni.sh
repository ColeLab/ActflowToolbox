#!/bin/bash
# Taku Ito
# 05/07/18

#########################
basedir="/projects3/NetworkDiversity/"
datadir="${basedir}/data/hcppreprocessedmsmall/"
hcpdir=/projects3/ExternalDatasets2/HCPData2/HCPS1200MSMAll/

#########################
### New subjects for QC'd 352 HCP subjs
subjNums1="100206 108020 117930 126325 133928 143224 153934 164636 174437 183034 194443 204521 212823 268749 322224 385450 463040 529953 587664 656253 731140 814548 877269 978578 100408 108222 118124 126426 134021 144832 154229 164939 175338 185139 194645 204622 213017 268850 329844 389357 467351 530635 588565 657659 737960 816653 878877 987074 101006 110007 118225 127933 134324 146331 154532 165638 175742 185341 195445 205119 213421 274542 341834 393247 479762 545345 597869 664757 742549 820745 887373 989987 102311 111009 118831 128632 135528 146432 154936 167036 176441 186141 196144 205725 213522 285345 342129 394956 480141 552241 598568 671855 744553 826454 896879 990366 102513 112516 118932 129028 135629 146533 156031 167440 176845 187850 196346 205826 214423 285446 348545 395756 481042 553344 599671 675661 749058 832651 899885 991267 102614 112920 119126 129129 135932 147636 157336 168745 177645 188145 198350 208226 214726 286347 349244 406432 486759 555651 604537 679568 749361 835657 901442 992774 103111 113316 120212 130013 136227 148133 157437 169545 178748 188549 198451 208327 217429 290136 352738 414229 497865 559457 615744 679770 753150 837560 907656 993675 103414 113619 120414 130114 136833 150726 157942 171330"
subjNums2="178950 189450 199453 209228 220721 298455 356948 419239 499566 561444 618952 680452 757764 841349 908860 103818 113922 121618 130619 137229 151829 158035 171633 179346 190031 200008 210112 221319 299154 361234 424939 500222 570243 622236 687163 769064 845458 911849 104416 114217 122317 130720 137532 151930 159744 172029 180230 191235 200614 211316 228434 300618 361941 432332 513130 571144 623844 692964 773257 857263 926862 105014 114419 122822 130821 137633 152427 160123 172938 180432 192035 200917 211417 239944 303119 365343 436239 513736 579665 638049 702133 774663 865363 930449 106521 114823 123521 130922 137936 152831 160729 173334 180533 192136 201111 211619 249947 305830 366042 436845 516742 580650 645450 715041 782561 871762 942658 106824 117021 123925 131823 138332 153025 162026 173536 180735 192439 201414 211821 251833 310621 371843 445543 519950 580751 647858 720337 800941 871964 955465 107018 117122 125222 132017 138837 153227 162329 173637 180937 193239 201818 211922 257542 314225 378857 454140 523032 585862 654350 725751 803240 872562 959574 107422 117324 125424 133827 142828 153631 164030 173940 182739 194140 202719 212015 257845 316633 381543 459453 525541 586460 654754 727553 812746 873968 966975"

subjNums="100206"

##
for subj in $subjNums1
do
    echo "Creating gray, white, ventricle, whole brain masks for subject ${subj}..."

    subjdir=${datadir}/${subj}/
    subjmaskdir=${subjdir}/masks
    functionalVolumeData=${hcpdir}/${subj}/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR.nii.gz

    if [ ! -e $subjmaskdir ]; then mkdir $subjmaskdir; fi 

    # HCP standard to parcel out white v gray v ventricle matter
    segparc=${hcpdir}/${subj}/MNINonLinear/wmparc.nii.gz

    # Change to subjmaskdir
    pushd $subjmaskdir

    ###############################
    ### Create whole brain masks
    echo "Creating whole brain mask for subject ${subj}..."
    3dcalc -overwrite -a $segparc -expr 'ispositive(a)' -prefix ${subj}_wholebrainmask.nii.gz
    # Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_wholebrainmask.nii.gz -prefix ${subj}_wholebrainmask_func.nii.gz
    # Dilate mask by 1 functional voxel (just in case the resampled anatomical mask is off by a bit)
    3dLocalstat -overwrite -nbhd 'SPHERE(-1)' -stat 'max' -prefix ${subj}_wholebrainmask_func_dil1vox.nii.gz ${subj}_wholebrainmask_func.nii.gz
    
   

    ###############################
    ### Create gray matter masks
    echo "Creating gray matter masks for subject ${subj}..." 
    # Indicate the mask value set for wmparc.nii.gz
    # Gray matter mask set
    maskValSet="8 9 10 11 12 13 16 17 18 19 20 26 27 28 47 48 49 50 51 52 53 54 55 56 58 59 60 96 97 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_gmMask.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_gmMask.nii.gz -prefix ${subj}_gmMask_func.nii.gz
    #Dilate mask by 1 functional voxel (just in case the resampled anatomical mask is off by a bit)
    3dLocalstat -overwrite -nbhd 'SPHERE(-1)' -stat 'max' -prefix ${subj}_gmMask_func_dil1vox.nii.gz ${subj}_gmMask_func.nii.gz
    
    rm -f ${subj}mask_temp.nii.gz
       
   
    
    ###############################
    ### Create white matter masks
    echo "Creating white matter masks for subject ${subj}..."

    # Indicate the mask value set for wmparc.nii.gz
    # White matter mask set
    maskValSet="250 251 252 253 254 255 3000 3001 3002 3003 3004 3005 3006 3007 3008 3009 3010 3011 3012 3013 3014 3015 3016 3017 3018 3019 3020 3021 3022 3023 3024 3025 3026 3027 3028 3029 3030 3031 3032 3033 3034 3035 4000 4001 4002 4003 4004 4005 4006 4007 4008 4009 4010 4011 4012 4013 4014 4015 4016 4017 4018 4019 4020 4021 4022 4023 4024 4025 4026 4027 4028 4029 4030 4031 4032 4033 4034 4035 5001 5002"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_wmMask.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_wmMask.nii.gz -prefix ${subj}_wmMask_func.nii.gz
    #Subtract graymatter mask from white matter mask (avoiding negative #s)
    3dcalc -a ${subj}_wmMask_func.nii.gz -b ${subj}_gmMask_func_dil1vox.nii.gz -expr 'step(a-b)' -prefix ${subj}_wmMask_func_eroded.nii.gz -overwrite
    
    rm -f ${subj}mask_temp.nii.gz
          

    
    ###############################
    ### Create ventricle masks
    echo "Creating ventricle matter masks for subject ${subj}..."

    # Indicate the mask value set for wmparc.nii.gz
    # Ventricle mask set
    maskValSet="4 43 14 15"

    # Add segments to mask
    maskNum=1
    for maskval in $maskValSet
    do
	if [ ${maskNum} = 1 ]; then
            3dcalc -a $segparc -expr "equals(a,${maskval})" -prefix ${subj}mask_temp.nii.gz -overwrite
        else
            3dcalc -a $segparc -b ${subj}mask_temp.nii.gz -expr "equals(a,${maskval})+b" -prefix ${subj}mask_temp.nii.gz -overwrite
        fi
	let maskNum++
    done
    #Make mask binary
    3dcalc -a ${subj}mask_temp.nii.gz -expr 'ispositive(a)' -prefix ${subj}_ventricles.nii.gz -overwrite
    #Resample to functional space
    3dresample -overwrite -master ${functionalVolumeData} -inset ${subj}_ventricles.nii.gz -prefix ${subj}_ventricles_func.nii.gz
    #Subtract graymatter mask from ventricles (avoiding negative #s)
    3dcalc -a ${subj}_ventricles_func.nii.gz -b ${subj}_gmMask_func_dil1vox.nii.gz -expr 'step(a-b)' -prefix ${subj}_ventricles_func_eroded.nii.gz -overwrite
    rm -f ${subjNum}mask_temp.nii.gz
    
    rm -f ${subj}mask_temp.nii.gz
          
    popd

done 



