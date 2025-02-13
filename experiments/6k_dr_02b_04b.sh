# Template for running experiments.

# meta options
save_dir='/scratch/kingb12/' 
model_name='6k_dr_02b_04b'
# common adjustments
max_epochs=200
learning_rate=0.00001
num_samples=7
max_sample_length=25
##################################################### Model Run Options #############################################
# Dataset options
enc_inputs='../data/rl_no_ing_15_6k_tr.th7'
dec_inputs='../data/rl_dec_inputs_25_6k_tr.th7'
outputs='../data/rl_outputs_25_6k_tr.th7'
in_lengths='../data/rl_in_lengths.th7'
out_lengths='../data/rl_out_lengths_25.th7'
helper='../data/rl_helper_6k_tr.th7'

valid_enc_inputs='/homes/iws/kingb12/data/rl_vno_ing_15_6k_tr.th7'
valid_dec_inputs='/homes/iws/kingb12/data/rl_vdec_inputs_25_6k_tr.th7'
valid_outputs='/homes/iws/kingb12/data/rl_voutputs_25_6k_tr.th7'
valid_in_lengths='/homes/iws/kingb12/data/rl_vin_lengths.th7'
valid_out_lengths='/homes/iws/kingb12/data/rl_vout_lengths_25.th7'

test_enc_inputs='/homes/iws/kingb12/data/rl_tno_ing_15_6k_tr.th7'
test_dec_inputs='/homes/iws/kingb12/data/rl_tdec_inputs_25_6k_tr.th7'
test_outputs='/homes/iws/kingb12/data/rl_toutputs_25_6k_tr.th7'
test_in_lengths='/homes/iws/kingb12/data/rl_tin_lengths.th7'
test_out_lengths='/homes/iws/kingb12/data/rl_tout_lengths_25.th7'

max_in_len=15
max_out_len=25
min_out_len=1
min_in_len=1
batch_size=4

init_enc_from=''
init_dec_from=''
wordvec_size=300
hidden_size=800
enc_dropout=0.2
dec_dropout=0.4
enc_dropout_loc='both'
dec_dropout_loc='both'
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
-enc_dropout_loc $enc_dropout_loc \
-dec_dropout_loc $dec_dropout_loc \
-enc_inputs $enc_inputs \
-dec_inputs $dec_inputs \
-outputs $outputs \
-in_lengths $in_lengths \
-out_lengths $out_lengths \
-helper $helper \
-valid_enc_inputs $valid_enc_inputs \
-valid_dec_inputs $valid_dec_inputs \
-valid_outputs $valid_outputs \
-valid_in_lengths $valid_in_lengths \
-valid_out_lengths $valid_out_lengths \
-max_in_len $max_in_len \
-max_out_len $max_out_len \
-min_out_len $min_out_len \
-min_in_len $min_in_len \
-batch_size $batch_size \
-wordvec_size $wordvec_size \
-hidden_size $hidden_size \
-enc_dropout $enc_dropout \
-dec_dropout $dec_dropout \
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
-monitor_outputs \
-gpu \
-stop_criteria_num_epochs $stop_criteria_num_epochs
