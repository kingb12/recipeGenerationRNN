# Template for running experiments.

# meta options
save_dir='/scratch/kingb12/' 
model_name='encdec_noing15_6k_trV_300_800_04drb_2'
# common adjustments
max_epochs=200
learning_rate=0.00001
num_samples=7
max_sample_length=25
##################################################### Model Run Options #############################################
# Dataset options
data_config='./experiments/encdec_noing15_6kV_300_800_drb_cfg.json'
max_in_len=15
max_out_len=25
min_out_len=1
min_in_len=1
batch_size=4

init_enc_from=''
init_dec_from=''
wordvec_size=300
hidden_size=800
dropout=0.4
dropout_loc='both'
num_enc_layers=1
num_dec_layers=1
weights=''

lr_decay=0.0
algorithm='adam'

print_loss_every=1000
save_prefix=$save_dir$model_name'/'$model_name
backup_save_dir=''
print_acc_every=0
print_examples_every=0
valid_loss_every=1
stop_criteria_num_epochs=2


######################################################### Evaluation Options ##############################################

max_gen_example_length=10
out=$save_prefix'.json'

######################################################## Actually Running #################################################

cd ..
mkdir $save_dir'/'$model_name

th EncoderDecoder.lua \
-dropout_loc $dropout_loc \
-data_config $data_config \
-max_in_len $max_in_len \
-max_out_len $max_out_len \
-min_out_len $min_out_len \
-min_in_len $min_in_len \
-batch_size $batch_size \
-wordvec_size $wordvec_size \
-hidden_size $hidden_size \
-dropout $dropout \
-num_enc_layers $num_enc_layers \
-num_dec_layers $num_dec_layers \
-weights $weights \
-no_average_loss $no_average_loss \
-max_epochs $max_epochs \
-learning_rate $learning_rate \
-lr_decay $lr_decay \
-algorithm $algorithm \
-print_loss_every $print_loss_every \
-save_model_at_epoch \
-save_prefix $save_prefix \
-backup_save_dir $backup_save_dir \
-run $run \
-print_acc_every $print_acc_every \
-print_examples_every $print_examples_every \
-valid_loss_every $valid_loss_every \
-run  \
-monitor_enc_states  \
-gpu \
-stop_criteria_num_epochs $stop_criteria_num_epochs 
