from recbole.quick_start import run_recbole

parameter_dict = {
   'train_neg_sample_args': None,
}
run_recbole(model='SASRecF', dataset='ml-100k', config_file_list=['test.yaml'], config_dict=parameter_dict)
