#!/bin/bash
###########################
#
#Scritp that reads parameter file and
#runs the integration routine sequentially for each parameter
#in the file
#
#
###########################

#Readibg parameter file

param_arr=$(awk -F= '{print $1}' parameters)
jobname="T_wdep_fit_point_FS_dense"  #JOBNAME importan to declare -has to be descriptive

#General info about the job
date_in="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_in}" >inforun
echo '....Jobs started running' >>  inforun

#Temporary directories where the runs will take place
dire_to_temps="../temp/temp_${jobname}_${date_in}"
rm -rf "${dire_to_temps}"
mkdir "${dire_to_temps}"

#loop over the parameters
for param_val in ${param_arr[@]}; do

	#create one temporary directory per parameter
	dire=""${dire_to_temps}"/${jobname}_${param_val}"
	rm -rf "${dire}"
	mkdir -vp "${dire}"


    cp IntSelfEnergy.py "${dire}"
    cp Lattice.py  "${dire}"
    cp StructureFactor.py  "${dire}"
    cp Dispersion.py  "${dire}"
	cp parameters "${dire}"
	cp runJ.sh "${dire}"
	#entering the temp directory, running and coming back
	cd "${dire}"
    mkdir ImgsRun
    mkdir DataRun

	nohup time python3 -u IntSelfEnergy.py 1 1 ${param_val} CH1 >> output.out 
	cd "../../../Modular"
	sleep 1

done

wait

#general info about the job as it ends
date_fin="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_fin}" >>inforun
echo 'Jobs finished running'>>inforun

#moving files to the data directory and tidying up
dire_to_data="../data/${jobname}_${date_fin}"
mkdir "${dire_to_data}"
mv "${dire_to_temps}"/* "${dire_to_data}"
mv inforun "${dire_to_data}"
rm -r "${dire_to_temps}"
