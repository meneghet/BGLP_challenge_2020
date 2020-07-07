#!/bin/bash

for patN in {'540','544','552','559','563','567','570','575','584','588','591','596'}
# for patN in '559'
do
	for PH in {'30','60'}
    # for PH in '30'
    do
        # file name
        fname='j_'$patN'_'$PH'.job'
        echo $fname
        
        # make new file
        echo "#!/bin/bash" > $fname
        # write qsub options
        echo "#$ -cwd" >> $fname
        echo "#$ -q Q@runner-03" >> $fname
        # write activate conda
        echo "source /nfsd/opt/anaconda3/anaconda3.sh" >> $fname
        echo "conda activate tensorflow" >> $fname
        # write script to be called
        echo cd .. >> $fname
        echo python -u main_test_after_selection_only_CGM.py $patN $PH >> $fname
        
        # qsub the file
        qsub $fname
    done
done