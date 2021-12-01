import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory


def tkinter_input():
    win = tk.Tk()
    win.title("Validation GUI")  # Add title
    win.geometry('480x300')

    lm = 2
    normal_wide = 10
    style = ttk.Style()
    style.configure("test.TButton", background="white", foreground="blue")

    def clickMe():
        nonlocal data_type, factor1, v_path, norm_thres0, norm_thres1, norm_thres2, \
            loss0, loss1, loss2, label_tag0, label_tag1, label_tag2, iso_z_sub_factor, is_denoise, is_sr, is_iso
        data_type = data_type_Chosen.get()

        factor1 = factor_Chosen1.get()
        # archi11 = archi1_Chosen1.get()
        # archi20 = archi2_Chosen0.get()
        # archi21 = archi2_Chosen1.get()
        # archi22 = archi2_Chosen2.get()
        loss0 = loss_Chosen0.get()
        loss1 = loss_Chosen1.get()
        loss2 = loss_Chosen2.get()
        norm_thres0 = norm_thres_entered0.get()
        norm_thres1 = norm_thres_entered1.get()
        norm_thres2 = norm_thres_entered2.get()
        label_tag0 = label_tag_entered0.get()
        label_tag1 = label_tag_entered1.get()
        label_tag2 = label_tag_entered2.get()
        iso_z_sub_factor = iso_z_sub_factor_entered.get()
        v_path = v_path_Choose.get()

        is_denoise = is_denoise.get()
        is_sr = is_sr.get()
        is_iso = is_iso.get()
        win.destroy()

    action = ttk.Button(win, text="Start running", command=clickMe, style="test.TButton")
    action.grid(column=1 + lm, row=15, columnspan=lm)  # Set its position in the interface

    ttk.Label(win, text="Choose data type:").grid(column=0, row=0, sticky=tk.W)  # Add a label
    data_type = tk.StringVar()
    data_type_Chosen = ttk.Combobox(win, width=40, textvariable=data_type, state='readonly')
    data_type_Chosen['values'] = ('Mitochondrial_inner_membrane', 'Microtubule', 'Intermediate_filament', 'ER',
                                  'Mitochondrial_matrix', 'Actin', 'Mitochondrial_outer_membrane',
                                  'Late_endosome')  # Sets the value of the drop-down list
    data_type_Chosen.grid(column=1, row=0, columnspan=3 * lm + 1)
    # Set its position in the interface.
    data_type_Chosen.current(3)
    # Set the default value displayed in the drop-down list. 3 is the subscript value of numberchosen ['values']

    ttk.Label(win, text="Choose net type:").grid(column=0, row=1, sticky=tk.W)
    ttk.Label(win, text="Denoise").grid(column=1, row=1, sticky=tk.N)
    ttk.Label(win, text="SR").grid(column=1 + lm, row=1, columnspan=lm, sticky=tk.N)
    ttk.Label(win, text="ISO").grid(column=2 + 2 * lm, row=1, columnspan=lm, sticky=tk.N)

    ttk.Label(win, text="Choose factor:").grid(column=0, row=2, sticky=tk.W)
    ttk.Label(win, text="1").grid(column=1, row=2, columnspan=lm, sticky=tk.N)
    ttk.Label(win, text="1").grid(column=2 + 2 * lm, row=2, columnspan=lm, sticky=tk.N)

    factor1 = tk.StringVar()
    factor_Chosen1 = ttk.Combobox(win, width=normal_wide, textvariable=factor1, state='readonly')
    factor_Chosen1['values'] = (2, 4)
    factor_Chosen1.grid(column=1 + lm, row=2, columnspan=lm)
    factor_Chosen1.current(0)

    # ttk.Label(win, text="Choose first net:").grid(column=0, row=3, sticky=tk.W)
    # ttk.Label(win, text="None").grid(column=1, row=3, columnspan=lm, sticky=tk.N)
    # ttk.Label(win, text="None").grid(column=2 + 2 * lm, row=3, columnspan=lm, sticky=tk.N)
    #
    # archi11 = tk.StringVar()
    # archi1_Chosen1 = ttk.Combobox(win, width=normal_wide, textvariable=archi11, state='readonly')
    # archi1_Chosen1['values'] = (None, 'dbpn3d', 'convnet3d')
    # archi1_Chosen1.grid(column=1 + lm, row=3, columnspan=lm, sticky=tk.N)
    # archi1_Chosen1.current(1)
    #
    # ttk.Label(win, text="Choose second net:").grid(column=0, row=4, sticky=tk.W)
    # archi20 = tk.StringVar()
    # archi2_Chosen0 = ttk.Combobox(win, width=normal_wide, textvariable=archi20, state='readonly')
    # archi2_Chosen0['values'] = ('drunet3d', 'unet3d')
    # archi2_Chosen0.grid(column=1, row=4, columnspan=lm, sticky=tk.W)
    # archi2_Chosen0.current(0)
    #
    # archi21 = tk.StringVar()
    # archi2_Chosen1 = ttk.Combobox(win, width=normal_wide, textvariable=archi21, state='readonly')
    # archi2_Chosen1['values'] = ('rdn3d')
    # archi2_Chosen1.grid(column=1 + lm, row=4, columnspan=lm)
    # archi2_Chosen1.current(0)
    #
    # archi22 = tk.StringVar()
    # archi2_Chosen2 = ttk.Combobox(win, width=normal_wide, textvariable=archi22, state='readonly')
    # archi2_Chosen2['values'] = ('drunet2d', 'unet2d')
    # archi2_Chosen2.grid(column=2 + 2 * lm, row=4, columnspan=lm)
    # archi2_Chosen2.current(0)

    ttk.Label(win, text="Choose loss fuction:").grid(column=0, row=5, sticky=tk.W)
    loss0 = tk.StringVar()
    loss_Chosen0 = ttk.Combobox(win, width=normal_wide, textvariable=loss0, state='readonly')
    loss_Chosen0['values'] = ('mse', 'mae')
    loss_Chosen0.grid(column=1, row=5, columnspan=lm, sticky=tk.W)
    loss_Chosen0.current(0)

    loss1 = tk.StringVar()
    loss_Chosen1 = ttk.Combobox(win, width=normal_wide, textvariable=loss1, state='readonly')
    loss_Chosen1['values'] = ('mse', 'mae')
    loss_Chosen1.grid(column=1 + lm, row=5, columnspan=lm)
    loss_Chosen1.current(0)

    loss2 = tk.StringVar()
    loss_Chosen2 = ttk.Combobox(win, width=normal_wide, textvariable=loss2, state='readonly')
    loss_Chosen2['values'] = ('mse', 'mae')
    loss_Chosen2.grid(column=2 + 2 * lm, row=5, columnspan=lm)
    loss_Chosen2.current(0)

    tk.Label(win, text="Normalize threshold:").grid(column=0, row=6, sticky=tk.W)
    norm_thres0 = tk.StringVar()
    norm_thres0.set('99.99')
    norm_thres_entered0 = ttk.Entry(win, width=12, textvariable=norm_thres0)
    norm_thres_entered0.grid(column=1, row=6, columnspan=lm, sticky=tk.W)
    norm_thres_entered0.focus()

    norm_thres1 = tk.StringVar()
    norm_thres1.set('99.99')
    norm_thres_entered1 = ttk.Entry(win, width=14, textvariable=norm_thres1)
    norm_thres_entered1.grid(column=1 + lm, row=6, columnspan=lm, sticky=tk.N)

    norm_thres2 = tk.StringVar()
    norm_thres2.set('99.99')
    norm_thres_entered2 = ttk.Entry(win, width=12, textvariable=norm_thres2)
    norm_thres_entered2.grid(column=2 + 2 * lm, row=6, columnspan=lm, sticky=tk.N)

    tk.Label(win, text="Label tag :").grid(column=0, row=7, sticky=tk.W)
    label_tag0 = tk.StringVar()
    label_tag0.set('confocal')
    label_tag_entered0 = ttk.Entry(win, width=12, textvariable=label_tag0)
    label_tag_entered0.grid(column=1, row=7, columnspan=lm, sticky=tk.W)
    label_tag_entered0.focus()

    label_tag1 = tk.StringVar()
    label_tag1.set('lightsheet')
    label_tag_entered1 = ttk.Entry(win, width=14, textvariable=label_tag1)
    label_tag_entered1.grid(column=1 + lm, row=7, columnspan=lm, sticky=tk.N)

    label_tag2 = tk.StringVar()
    label_tag2.set('lightsheet')
    label_tag_entered2 = ttk.Entry(win, width=12, textvariable=label_tag2)
    label_tag_entered2.grid(column=2 + 2 * lm, row=7, columnspan=lm, sticky=tk.N)

    tk.Label(win, text="ISO z sub factor:").grid(column=0, row=10, sticky=tk.W)
    iso_z_sub_factor = tk.StringVar()
    iso_z_sub_factor.set('1.0')
    iso_z_sub_factor_entered = ttk.Entry(win, width=12, textvariable=iso_z_sub_factor)
    iso_z_sub_factor_entered.grid(column=6, row=10, sticky=tk.E)

    def select_V_Path():
        path_hr = askdirectory(title="Please choose the Validation data path")
        v_path.set(path_hr)

    tk.Label(win, text="Validation data path:").grid(column=0, row=11, sticky=tk.E)
    v_path = tk.StringVar()
    v_path_Choose = ttk.Entry(win, width=28, textvariable=v_path)
    v_path_Choose.grid(column=1, row=11, columnspan=4)
    v_path_Choose_button = ttk.Button(win, text="Choose", command=select_V_Path, style="test.TButton")
    v_path_Choose_button.grid(column=6, row=11, sticky=tk.E)

    size_row = 12
    is_denoise = tk.IntVar()
    check1 = tk.Checkbutton(win, text="Denoise", variable=is_denoise)
    check1.select()
    check1.grid(column=1, row=size_row, sticky=tk.W)
    is_sr = tk.IntVar()
    check2 = tk.Checkbutton(win, text="SR", variable=is_sr)
    check2.select()
    check2.grid(column=1 + lm, row=size_row, sticky=tk.E)
    is_iso = tk.IntVar()
    check3 = tk.Checkbutton(win, text="ISO", variable=is_iso)
    check3.select()
    check3.grid(column=2 + 2 * lm, row=size_row, sticky=tk.E)
    data_type_Chosen.bind("<Return>", clickMe)
    win.mainloop()
    return data_type, factor1, v_path, norm_thres0, norm_thres1, norm_thres2, \
            loss0, loss1, loss2, label_tag0, label_tag1, label_tag2, iso_z_sub_factor, is_denoise, is_sr, is_iso


data_type, factor1,  v_path, norm_thres0, norm_thres1, norm_thres2, loss0, loss1, \
loss2, label_tag0, label_tag1, label_tag2, iso_z_sub_factor, is_denoise, is_sr, is_iso = tkinter_input()

label_tag_denoise = label_tag0
label_tag_sr = label_tag1
label_tag_iso = label_tag2
data_type = data_type

z_sub_factor = float(iso_z_sub_factor)
factor_sr = int(factor1)
norm_thres_denoise = float(norm_thres0)
norm_thres_sr = float(norm_thres1)
norm_thres_iso = float(norm_thres2)


# archi1_sr = archi1_ if archi1_ != 'None' else None
archi1_sr = 'dbpn3d'

archi2_denoise = 'drunet3d'
archi2_sr = 'rdn3d'
archi2_iso = 'drunet2d'

loss_denoise = loss0
loss_sr = loss1
loss_iso = loss2

from easydict import EasyDict as edict

config_test_denoise = edict()
config_test_sr = edict()
config_test_iso = edict()
valid_using_batch_norm = False
valid_device_id = 0
valid_lr_img_path = v_path

"""
config for Denoise net
"""
config_test_denoise.factor = 1
config_test_denoise.archi2 = archi2_denoise
config_test_denoise.archi1 = None
config_test_denoise.loss = loss_denoise
config_test_denoise.valid_lr_img_size = [1, 60, 60, 60, 1]  # [batch, depth, height, width, channels]
config_test_denoise.valid_conv_kernel = 3
config_test_denoise.norm_thres = [norm_thres_denoise]
config_test_denoise.archi_str = '1stage_{}'.format(config_test_denoise.archi2)
config_test_denoise.label = '{}_{}_Denoise_{}_factor1_{}'.format(label_tag_denoise, data_type,
                                                                 config_test_denoise.archi_str, loss_denoise)
config_test_denoise.valid_ckpt_dir = "checkpoint/{}/".format(config_test_denoise.label)
config_test_denoise.valid_denoise_saving_path = "{}/{}/".format(valid_lr_img_path, config_test_denoise.label)

"""
config for SR net
"""
config_test_sr.factor = factor_sr
config_test_sr.archi2 = archi2_sr
config_test_sr.archi1 = archi1_sr
config_test_sr.loss = loss_sr
config_test_sr.valid_lr_img_size = [1, 60, 60, 60, 1]  # [batch, depth, height, width, channels]
config_test_sr.valid_conv_kernel = 3
config_test_sr.norm_thres = [norm_thres_sr]
config_test_sr.archi_str = '2stage_{}+{}'.format(config_test_sr.archi1, config_test_sr.archi2) if \
    config_test_sr.archi1 is not None else '1stage_{}'.format(config_test_sr.archi2)
config_test_sr.label = '{}_{}_SR_{}_factor{}_{}'.format(label_tag_sr, data_type,
                                                        config_test_sr.archi_str, config_test_sr.factor, loss_sr)
config_test_sr.valid_ckpt_dir = "checkpoint/{}/".format(config_test_sr.label)
config_test_sr.valid_sr_saving_path = "{}/{}/".format(valid_lr_img_path, config_test_sr.label)

"""
config for Iso net
"""
config_test_iso.factor = 1
config_test_iso.archi2 = archi2_iso
config_test_iso.archi1 = None  # [None, 'dbpn', 'convnet'] # None if 1stage
config_test_iso.loss = loss_iso
config_test_iso.valid_lr_img_size = [60, 60, 60, 1]  # [batch, height, width, channels]
config_test_iso.valid_conv_kernel = 3
config_test_iso.norm_thres = [norm_thres_iso]
config_test_iso.archi_str = '1stage_{}'.format(config_test_iso.archi2)
config_test_iso.label = '{}_{}_ISO_{}_factor1_{}'.format(label_tag_iso, data_type, config_test_iso.archi_str, loss_iso)
config_test_iso.valid_ckpt_dir = "checkpoint/{}/".format(config_test_iso.label)
config_test_iso.valid_iso_saving_path = "{}/{}/".format(valid_lr_img_path, config_test_iso.label)
