#!/bin/bash
###########################
#
#Scritp that reads parameter file and
#runs the integration routine for each parameter
#in the file
#
#
###########################

#Readibg parameter file

T_arr=$(awk -F= '{print $1}' Ts.dat)
jobname="temperature_dep_no_shift_at_zero_frequency_freqdep_integrandplot_log"  #JOBNAME importan to declare -has to be descriptive

#General info about the job
date_in="`date "+%Y-%m-%d-%H-%M-%S"`"
echo "${date_in}" >inforun
echo '....Jobs started running' >>  inforun

#Temporary directories where the runs will take place
dire_to_temps="../temp/temp_${jobname}_${date_in}"
rm -rf "${dire_to_temps}"
mkdir "${dire_to_temps}"

#loop over the parameters
for T_val in ${T_arr[@]}; do

	#create one temporary directory per parameter
	dire=""${dire_to_temps}"/${jobname}_${T_val}"
	rm -rf "${dire}"
	mkdir -vp "${dire}"


    cp shift_at_zero_frequency_fermi_surface_points_nofreqdep_integrandplot.py "${dire}"
	#entering the temp directory, running and coming back
	cd "${dire}"

	time python3 -u shift_at_zero_frequency_fermi_surface_points_nofreqdep.py ${T_val} 2000 >> output.out & 
	# time python3 -u shift_at_zero_frequency_fermi_surface_points.py ${T_val} 1000 >> output.out & 
	# time python3 -u shift_at_zero_frequency_fermi_surface_points_nofreqdep_integrandplot.py ${T_val} 1000 >> output.out & 
	cd "../../../MD_int"
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
