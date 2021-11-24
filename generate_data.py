from crop_data import generate_training_data

# import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from utils import plot_some
from tkinter import *
from tkinter.filedialog import askdirectory, askopenfilename
import tkinter as tk
from tkinter import ttk
import numpy as np


def tkinter_input():
    win = tk.Tk()
    win.title("Crop data")  # Add title
    win.geometry('480x360')

    lm = 7


    def clickMe():
        nonlocal factor, generate_mr, hr_path, lr_path, psf, patch_size_d, patch_size_h, patch_size_w, z_subsample, \
            thre, poisson_noise, gauss_noise
        factor = factor_Chosen.get()
        generate_mr = generate_mr.get()
        hr_path = hr_path_Choose.get()
        lr_path = lr_path_Choose.get()
        psf = psf_path_Choose.get()
        patch_size_d = patch_size_d_entered.get()
        patch_size_h = patch_size_h_entered.get()
        patch_size_w = patch_size_w_entered.get()
        z_subsample = z_subsample_entered.get()
        thre = thre_entered.get()
        poisson_noise = poisson_noise.get()
        gauss_noise = gauss_noise_entered.get()
        win.destroy()

    style = ttk.Style()
    style.configure("test.TButton", background="white", foreground="blue")
    action = ttk.Button(win, text="Start running", command=clickMe, style="test.TButton")
    action.grid(column=3, row=15, columnspan=2)

    Label(win, text="-------------------------------------Global parameters-------------------------------------").grid(
        column=0, row=0, columnspan=9, sticky=tk.N)

    def select_HR_Path():
        path_hr = askdirectory(title="Please choose the HR path")
        hr_path.set(path_hr)

    hr_path = tk.StringVar()
    Label(win, text="HR path:").grid(column=0, row=1, sticky=tk.N)
    hr_path_Choose = ttk.Entry(win, width=40, textvariable=hr_path)
    hr_path_Choose.grid(column=1, row=1, columnspan=lm, sticky=tk.W)
    hr_path_Choose_button = ttk.Button(win, text="Choose", command=select_HR_Path, style="test.TButton")
    hr_path_Choose_button.grid(column=7, row=1, sticky=tk.W)

    size_row = 2
    tk.Label(win, text="Patch size:").grid(column=0, row=size_row, sticky=tk.N)
    patch_size_d = tk.StringVar()
    patch_size_d.set('32')
    tk.Label(win, text="Depth:").grid(column=1, row=size_row, sticky=tk.W)
    patch_size_d_entered = ttk.Entry(win, width=5, textvariable=patch_size_d)
    patch_size_d_entered.grid(column=2, row=size_row, sticky=tk.W)

    tk.Label(win, text="Height:").grid(column=3, row=size_row, sticky=tk.W)
    patch_size_h = tk.StringVar()
    patch_size_h.set('32')
    patch_size_h_entered = ttk.Entry(win, width=5, textvariable=patch_size_h)
    patch_size_h_entered.grid(column=4, row=size_row, sticky=tk.W)

    tk.Label(win, text="Width:").grid(column=5, row=size_row, sticky=tk.W)
    patch_size_w = tk.StringVar()
    patch_size_w.set('32')
    patch_size_w_entered = ttk.Entry(win, width=5, textvariable=patch_size_w)
    patch_size_w_entered.grid(column=6, row=size_row, sticky=tk.W)

    tk.Label(win, text="Threshold:").grid(column=0, row=3, sticky=tk.N)
    thre = tk.StringVar()
    thre.set(0.9)
    thre_entered = ttk.Entry(win, width=5, textvariable=thre)
    thre_entered.grid(column=1, row=3, sticky=tk.W)

    tk.Label(win, text="Gaussian noise:").grid(column=2, row=3, columnspan=2, sticky=tk.W)
    gauss_noise = tk.StringVar()
    gauss_noise.set(0)
    gauss_noise_entered = ttk.Entry(win, width=5, textvariable=gauss_noise)
    gauss_noise_entered.grid(column=4, row=3, sticky=tk.W)

    poisson_noise = tk.IntVar()
    check2 = tk.Checkbutton(win, text="Possion noise", variable=poisson_noise)
    # check2.select()
    check2.grid(column=5, row=3, columnspan=2, sticky=tk.W)

    def select_LR_Path():
        path_lr = askdirectory(title="Please choose the LR path")
        lr_path.set(path_lr)

    Label(win, text="------------------------------------3D data parameters------------------------------------").grid(
        column=0, row=4, columnspan=9, sticky=tk.N)

    lr_path = tk.StringVar()
    row_2 = 5
    Label(win, text="LR path:").grid(column=0, row=row_2, sticky=tk.N)
    lr_path_Choose = ttk.Entry(win, width=40, textvariable=lr_path)
    lr_path_Choose.grid(column=1, row=row_2, columnspan=lm, sticky=tk.W)
    lr_path_Choose_button = ttk.Button(win, text="Choose", command=select_LR_Path, style="test.TButton")
    lr_path_Choose_button.grid(column=7, row=row_2, sticky=tk.E)

    ttk.Label(win, text="Choose factor:").grid(column=0, row=row_2+1, sticky=tk.W)
    factor = tk.StringVar()
    factor_Chosen = ttk.Combobox(win, width=5, textvariable=factor, state='readonly')
    factor_Chosen['values'] = (1, 2)
    factor_Chosen.grid(column=1, row=row_2+1, columnspan=2, sticky=tk.W)
    factor_Chosen.current(1)

    generate_mr = tk.IntVar()
    check1 = tk.Checkbutton(win, text="Generate mr", variable=generate_mr)
    check1.select()
    check1.grid(column=3, row=row_2+1, columnspan=2, sticky=tk.W)

    Label(win, text="--------------------------------------ISO parameters--------------------------------------").grid(
        column=0, row=7, columnspan=9, sticky=tk.N)

    def select_psf():
        psf_ = askopenfilename(title="Select txt file", filetypes=(("txt files", "*.txt"),))
        psf.set(psf_)

    psf = tk.StringVar()
    row_3 = 8
    psf.set(None)
    Label(win, text="PSF file:").grid(column=0, row=row_3, sticky=tk.N)
    psf_path_Choose = ttk.Entry(win, width=40, textvariable=psf)
    psf_path_Choose.grid(column=1, row=row_3, columnspan=lm, sticky=tk.W)
    psf_path_Choose_button = ttk.Button(win, text="Choose", command=select_psf, style="test.TButton")
    psf_path_Choose_button.grid(column=7, row=row_3, sticky=tk.W)

    tk.Label(win, text="Axis subsample:").grid(column=2, row=row_3+1, columnspan=2, sticky=tk.E)
    z_subsample = tk.StringVar()
    z_subsample.set(1.0)
    z_subsample_entered = ttk.Entry(win, width=5, textvariable=z_subsample)
    z_subsample_entered.grid(column=4, row=row_3+1, sticky=tk.W)

    factor_Chosen.bind("<Return>", clickMe)
    win.mainloop()
    return factor, generate_mr, hr_path, lr_path, psf, patch_size_d, patch_size_h, patch_size_w, z_subsample, \
           thre, poisson_noise, gauss_noise


factor, generate_mr, hr_path, lr_path, psf, patch_size_d, patch_size_h, patch_size_w, z_subsample, \
thre, poisson_noise, gauss_noise = tkinter_input()

psf_ = psf
d = int(patch_size_d)
h = int(patch_size_h)
w = int(patch_size_w)
hr_path = hr_path
lr_path = lr_path
train_img_size_lr = [h, w] if d == 1 else [d, h, w]  # [d,h,w] or [h,w]
factor = int(factor)
mr_need = generate_mr if factor > 1 else False
psf = None if psf_ == 'None' else np.loadtxt(psf_)
z_subsample = float(z_subsample)
threshold = float(thre)
poisson_noise = poisson_noise
gauss_sigma = float(gauss_noise)

hr, lr = generate_training_data(
    hr_path=hr_path,
    lr_path=lr_path,
    patch_size=train_img_size_lr,
    factor=factor,
    mr_need=mr_need,
    psf=psf,
    z_sub_sample=z_subsample,
    threshold=threshold,
    poisson_noise=poisson_noise,
    gauss_sigma=gauss_sigma,
)

if len(hr.shape) == 5:
    for i in range(1):
        plt.figure('xy_slice_%s' % str(i), figsize=(16, 4))
        sl = slice(8 * i, 8 * (i + 3), 3), 0, slice(None), slice(None)
        plot_some(hr[sl], lr[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
        plt.show()

    for i in range(1):
        plt.figure('yz_slice_%s' % str(i), figsize=(16, 2))
        sl = slice(8 * i, 8 * (i + 3), 3), slice(None), slice(None), 0
        plot_some(hr[sl], lr[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
        plt.show()
else:
    for i in range(2):
        plt.figure('yz_slice_%s' % str(i), figsize=(16, 4))
        sl = slice(8 * i, 8 * (i + 20), 20), slice(None), slice(None)
        plot_some(hr[sl], lr[sl], title_list=[np.arange(sl[0].start, sl[0].stop)])
        plt.show()
