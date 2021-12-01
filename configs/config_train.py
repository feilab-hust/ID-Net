import tkinter as tk
from tkinter import ttk


def tkinter_input():
    win = tk.Tk()
    win.title("Training GUI")  # Add title
    win.geometry('400x240')

    lm = 6

    style = ttk.Style()
    style.configure("test.TButton", background="white", foreground="blue")

    def clickMe():
        nonlocal data_type, net_type, factor, loss, label_tag, patch_size_d, patch_size_h, patch_size_w,using_lpips
        data_type = data_type_Chosen.get()
        net_type = net_type_Chosen.get()
        factor = factor_Chosen.get()
        # archi1 = archi1_Chosen.get()
        # archi2 = archi2_Chosen.get()
        loss = loss_Chosen.get()
        label_tag = label_tag_entered.get()
        patch_size_d = patch_size_d_entered.get()
        patch_size_h = patch_size_h_entered.get()
        patch_size_w = patch_size_w_entered.get()
        using_lpips = using_lpips.get()
        win.destroy()

    action = ttk.Button(win, text="Start running", command=clickMe, style="test.TButton")
    action.grid(column=3, row=15, columnspan=2)

    ttk.Label(win, text="Choose data type:").grid(column=0, row=0, sticky=tk.W)
    data_type = tk.StringVar()
    data_type_Chosen = ttk.Combobox(win, width=30, textvariable=data_type, state='readonly')
    data_type_Chosen['values'] = ('Mitochondrial_inner_membrane', 'Microtubule', 'Intermediate_filament', 'ER', 'Drp1',
                                  'Mitochondrial_matrix', 'Actin', 'Mitochondrial_outer_membrane',
                                  'Late_endosome')
    data_type_Chosen.grid(column=1, row=0, columnspan=lm)
    data_type_Chosen.current(3)

    ttk.Label(win, text="Choose net type:").grid(column=0, row=1, sticky=tk.W)
    net_type = tk.StringVar()
    net_type_Chosen = ttk.Combobox(win, width=30, textvariable=net_type, state='readonly')
    net_type_Chosen['values'] = ('SR', 'Denoise', 'ISO')
    net_type_Chosen.grid(column=1, row=1, columnspan=lm)
    net_type_Chosen.current(0)

    ttk.Label(win, text="Choose factor:").grid(column=0, row=2, sticky=tk.W)
    factor = tk.StringVar()
    factor_Chosen = ttk.Combobox(win, width=30, textvariable=factor, state='readonly')
    factor_Chosen['values'] = (1, 2, 4)
    factor_Chosen.grid(column=1, row=2, columnspan=lm)
    factor_Chosen.current(1)

    # ttk.Label(win, text="Choose first net:").grid(column=0, row=3, sticky=tk.W)
    # archi1 = tk.StringVar()
    # archi1_Chosen = ttk.Combobox(win, width=30, textvariable=archi1, state='readonly')
    # archi1_Chosen['values'] = (None, 'dbpn3d', 'convnet3d')
    # archi1_Chosen.grid(column=1, row=3, columnspan=lm)
    # archi1_Chosen.current(0)
    #
    # ttk.Label(win, text="Choose second net:").grid(column=0, row=4, sticky=tk.W)
    # archi2 = tk.StringVar()
    # archi2_Chosen = ttk.Combobox(win, width=30, textvariable=archi2, state='readonly')
    # archi2_Chosen['values'] = ('rdn3d', 'unet3d', 'dbpn3d', 'drunet3d', 'convnet3d', 'unet2d', 'drunet2d')
    # archi2_Chosen.grid(column=1, row=4, columnspan=lm)
    # archi2_Chosen.current(0)

    ttk.Label(win, text="Choose loss fuction:").grid(column=0, row=5, sticky=tk.W)
    loss = tk.StringVar()
    loss_Chosen = ttk.Combobox(win, width=15, textvariable=loss, state='readonly')
    loss_Chosen['values'] = ('mse', 'mae')
    loss_Chosen.grid(column=1, row=5, columnspan=lm-1, sticky=tk.N)
    loss_Chosen.current(0)

    using_lpips = tk.IntVar()
    check1 = tk.Checkbutton(win, text="LPIPS loss", variable=using_lpips)
    check1.grid(column=5, row=5, columnspan=2, sticky=tk.E)

    tk.Label(win, text="Label tag :").grid(column=0, row=9)
    label_tag = tk.StringVar()
    label_tag.set("lightsheet")
    label_tag_entered = ttk.Entry(win, width=30, textvariable=label_tag)
    label_tag_entered.grid(column=1, row=9, columnspan=lm)
    label_tag_entered.focus()

    size_row = 11
    tk.Label(win, text="Patch size :").grid(column=0, row=size_row)
    patch_size_d = tk.StringVar()
    patch_size_d.set('32')
    tk.Label(win, text="Depth:").grid(column=1, row=size_row)
    patch_size_d_entered = ttk.Entry(win, width=5, textvariable=patch_size_d)
    patch_size_d_entered.grid(column=2, row=size_row)

    tk.Label(win, text="Height:").grid(column=3, row=size_row)
    patch_size_h = tk.StringVar()
    patch_size_h.set('32')
    patch_size_h_entered = ttk.Entry(win, width=5, textvariable=patch_size_h)
    patch_size_h_entered.grid(column=4, row=size_row)

    tk.Label(win, text="Width:").grid(column=5, row=size_row)
    patch_size_w = tk.StringVar()
    patch_size_w.set('32')
    patch_size_w_entered = ttk.Entry(win, width=5, textvariable=patch_size_w)
    patch_size_w_entered.grid(column=6, row=size_row)

    data_type_Chosen.bind("<Return>", clickMe)
    win.mainloop()
    return data_type, net_type, factor, loss, label_tag, patch_size_d, patch_size_h, patch_size_w,using_lpips


data_type, net_type, factor, loss, label_tag, patch_size_d, patch_size_h, patch_size_w, using_lpips = tkinter_input()

label_tag = label_tag
data_type = data_type
net_type = net_type

if net_type == 'SR':
    factor = int(factor)
    archi1 = 'dbpn3d'
    archi2 = 'rdn3d'

elif net_type == 'Denoise':
    factor = 1
    archi1 = None
    archi2 = 'drunet3d'

elif net_type == 'ISO':
    factor = 1
    archi1 = None
    archi2 = 'drunet2d'



loss = loss
using_lpips = using_lpips
d = int(patch_size_d)
h = int(patch_size_h)
w = int(patch_size_w)

train_img_size_lr = [h, w, 1] if d == 1 else [d, h, w, 1]  # [d,h,w] or [h,w]
train_img_size_hr = [factor * h, factor * w, 1] if d == 1 else [factor * d, factor * h, factor * w, 1]

archi_str = '2stage_{}+{}'.format(archi1, archi2) if archi1 is not None else '1stage_{}'.format(archi2)
label = '{}_{}_{}_{}_factor{}_{}'.format(label_tag, data_type, net_type, archi_str, factor, loss)

using_batch_norm = False

train_test_data_path = None
train_valid_lr_path = None  # "data/bead_simu/valid_otf/"   # valid on_the_fly
