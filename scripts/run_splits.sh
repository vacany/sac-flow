# Description: Run the planned experiments in parallel on the GPU
for gpu in {0..7}
do
  nohup python scripts/run_planned_experiments.py $gpu >>logs/$gpu &
done
