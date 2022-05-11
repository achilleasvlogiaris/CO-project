import os

window_name = 'window_tfold[0,40000]'
input_type = ['corrtrain', 'spec corrtrain']
dataset = "ASCAD_byte_2_25000_65000_40000f:Profiling_traces"
# number_of_hidden_layers_list = [1, 2, 3, 4, 5]
number_of_hidden_layers_list = [2]
# num_epochs_list = [100, 200, 300, 400, 500]
num_epochs_list = [100]
# lr_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]
lr_list = [0.0001]
# batch_size_list = [64, 128, 256, 512, 1023]
batch_size_list = [512]

# os.system("./emma.py 'window[0,1500]' corrtrain ASCAD_40150_51349:Prof"
#                                         "iling_traces --tfold --n-hidden-layers 1 --epochs 100 "
#                                         "--lr 0.0001 --batch-size 512 --local")

for number_of_hidden_layers in number_of_hidden_layers_list:
    for epochs in num_epochs_list:
        for lr in lr_list:
            for batch_size in batch_size_list:
                os.system("./emma.py " + str(window_name) + " " + str(input_type[0]) + " " + dataset + " --tfold" + " --n-hidden-layers "
                          + str(number_of_hidden_layers) + " --epochs " + str(epochs) + " --lr " + str(lr) + " --batch-size "
                          + str(batch_size) + " --local")
            # for batch_size in batch_size_list:
            #     os.system("./emma.py " + str(window_name) + " " + str(input_type[0]) + " " + dataset + " --n-hidden-layers "
            #               + str(number_of_hidden_layers) + " --epochs " + str(epochs) + " --lr " + str(lr) + " --batch-size "
            #               + str(batch_size) + " --local")

                # os.system("./emma.py " + str(window_name) + " " + str(input_type[1]) + " " + dataset + " --tfold" + " --n-hidden-layers "
                #           + str(number_of_hidden_layers) + " --epochs " + str(epochs) + " --lr " + str(lr) + " --batch-size "
                #           + str(batch_size) + " --local")