def fit_tm(slm, tm):
    # define target
    target_shape = (int(tm.shape[0] ** 0.5), int(tm.shape[0] ** 0.5))

    tgt = phase_conjugation.Target(target_shape)

    x, y = (-10,  -10)

    target_frame = tgt.square((1, 1), offset_x=x, offset_y=y, intensity=1)
    # target_frame = tgt.gauss(num=16, order=0, w0=1e-4, slm_calibration_px=112)

    # phase conjugation - create mask
    msk = phase_conjugation.InverseLight(target_frame, tm, slm_macropixel=slm_macropixel, calib_px=112)
    phase_mask = msk.inverse_prop(conj=True)

    # merge phase mask into an slm pattern
    patternSLM = pt.PatternsBacic(resX, resY)
    focusing_mask = patternSLM.pattern_to_SLM(phase_mask, gray = 10)

    # apply mask
    slm.sendArray(focusing_mask)
    pass

def plot():
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    msk = axs[0].imshow(mask)
    axs[0].set_title("Optimized mask")
    axs[0].set_xlabel("SLM x-pixels #")
    axs[0].set_ylabel("SLM y-pixels #")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(msk, cax=cax)

    axs[1].plot(coeffs)
    axs[1].set_title("Optimized coefficients")
    axs[1].set_xlabel("Number of iterations #")
    axs[1].set_ylabel("Coefficients")
    axs[1].set_box_aspect(1)


    axs[2].plot(cost)
    axs[2].set_title("Cost function vs iterations")
    axs[2].set_xlabel("Number of iterations #")
    axs[2].set_ylabel("Cost function")
    axs[2].set_box_aspect(1)

    fig.tight_layout()

    figpath = zern.filepath + 'optim'
    plt.savefig(figpath, dpi=200, transparent=True)

    def filter_dict(dict, num, N):
        z0 = 1
        z1 = num
        keys =list(np.linspace(z0, z1, N).astype('uint8'))
        dict_new = {key:dict[key] for key in keys}
        return dict_new

    def plots(dict, mn=(3, 5), figsize=(8, 4), log=False):
        nrows, ncols = mn
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for (idx, matrix), ax in zip(dict.items(), axs.ravel()):
            if log:
                im = ax.imshow(matrix, norm=LogNorm())
            else:     
                im = ax.imshow(matrix)
            ax.axis('off')
            ax.set_title(r"$Z_{%d}$" %(idx))
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        fig.tight_layout()

        return fig