

for i in `seq 1 5`;
do 
  PYTHONPATH=. JUICER_CONFIG=../config/juicer-config-local.yaml python ../ajuda.py -c ../config/juicer-config-local.yaml -w 72 -j 342 -p multi  > log_time
done

for i in `seq 1 5`;
do      
  PYTHONPATH=. JUICER_CONFIG=../config/juicer-config-local.yaml python ../ajuda.py -c ../config/juicer-config-local.yaml -w 72 -j 343 -p spark  >> log_time
done

for i in `seq 1 5`;
do      
  PYTHONPATH=. JUICER_CONFIG=../config/juicer-config-local.yaml python ../ajuda.py -c ../config/juicer-config-local.yaml -w 72 -j 344 -p pandas >> log_time
done
