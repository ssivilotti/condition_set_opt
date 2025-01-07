#!/bin/bash

TEMPLATE_FILE=$(pwd)/"template_submit.bash"

PARENT_DIR=$(pwd)

current_date=$(date +"%Y-%m-%d")

current_time=$(date +"%H:%M:%S")

DATASETS=("buchwald_hartwig")
NUM_CPUS=2
BATCH_SIZE=40
OPT_INIT=(0 20 40 60 80)
N_REPEATS=20
LEARNER_TYPE=(-1 0 8 9)
MODEL_TYPE="RF"
STOCHASTIC_BATCH_NUM=-1
COND_SUBSET=(0 1 2 3 4 5)

#create a directory
JOB_DIR=$PARENT_DIR/runs/$current_date/$current_time

mkdir -p $PARENT_DIR/runs/$current_date
mkdir -p $JOB_DIR

count=0

cd $JOB_DIR

# for each job:
for SUBSET in ${COND_SUBSET[@]}; do
    for LEARNER in ${LEARNER_TYPE[@]}; do
        for INIT in ${OPT_INIT[@]}; do
            # for DATASET in ${DATASETS[@]}; do
            # edit template submit file with specific job parameters
            submit_name="${DATASETS}_${SUBSET}_${LEARNER}_${INIT}"
            submit_file=$JOB_DIR/"${submit_name}.bash"
            cp $TEMPLATE_FILE $submit_file
            sed -i "s/{name}/${submit_name}/g" ${submit_file}
            sed -i "s|{dir}|${PARENT_DIR}|g" ${submit_file}
            sed -i "s/{dataset}/${DATASETS}/g" $submit_file
            sed -i "s/{num_cpus}/${NUM_CPUS}/g" $submit_file
            sed -i "s/{batch_size}/${BATCH_SIZE}/g" $submit_file
            sed -i "s/{opt_init}/${INIT}/g" $submit_file
            sed -i "s/{n_repeats}/${N_REPEATS}/g" $submit_file
            sed -i "s/{learner_type}/${LEARNER}/g" $submit_file
            sed -i "s/{model_type}/${MODEL_TYPE}/g" $submit_file
            sed -i "s/{stochastic_batch_num}/${STOCHASTIC_BATCH_NUM}/g" $submit_file
            sed -i "s/{cond_subset}/${SUBSET}/g" $submit_file
            # submit job
            qsub $submit_file
            count=$((count+1))
            # done
        done
    done
done