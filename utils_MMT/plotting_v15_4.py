"""Plotting helpers used by the notebooks and the explainability figures.

Two families of plots live here:

1. **2-D structure plots** – grids of RDKit depictions
   (:func:`show_mols`, :func:`plot_molecules_from_list`,
   :func:`plot_smiles_from_df`, used e.g. in notebook 4.0).
2. **HSQC/COSY scatter overlays** – comparing real vs simulated 2-D spectra
   (:func:`plot_compare_scatter_plot`, the ``*_without_direction*`` variants).
   These expect dataframes with columns ``F2 (ppm)``, ``F1 (ppm)`` and, for
   simulated data, ``direction``/``Intensity``.

NOTE (known issue, left unchanged)
    本文件存在若干“看起来能用、实际会报错/画空白”的问题，逐一在函数附近标注：
      1. :func:`plot_smiles_from_df` 依赖未导入的 ``hf`` 与 ``display``，
         以及未定义的 ``pd``/``pd.concat``（虽然文件顶部没有 ``import pandas``）
         —— 运行必 ``NameError``。
      2. :func:`plot_compare_scatter_plot` 的 ``style`` 形参默认是 **list**
         ``["sim","orig","both"]``，但函数内用 ``style == "orig"`` 比较，
         永远为 ``False``，于是散点循环一次都不执行 → 只有空坐标轴。
      3. 多个函数对 ``df_orig`` / ``df_sim`` 做 **原地修改**（写入 ``Phase`` 列），
         调用方传入的 DataFrame 会被污染。
    以上均按约定只加注释，不修改实现。
"""


from rdkit.Chem.Draw import rdMolDraw2D
import math
import cairosvg
import os
from IPython.display import SVG
import matplotlib.pyplot as plt
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import base64
from io import BytesIO


#inspired by https://github.com/rdkit/UGM_2020/blob/master/Notebooks/Landrum_WhatsNew.ipynb
def show_mols(mols, mols_per_row = 5, size=200, min_font_size=12, legends=[], file_name=''):
    """Draw a grid of molecules as an inline SVG (optionally exported to a file).

    Args:
        mols (Sequence[Chem.Mol]): molecules to draw.
        mols_per_row (int): grid width; clamped to ``len(mols)`` so a short
            list does not leave empty columns.
        size (int): per-molecule pixel size (also the SVG viewport tile size).
        min_font_size (int): RDKit's minimum atom-label font size.
        legends (List[str]): optional per-molecule labels; must be at least as
            long as ``mols`` (otherwise the function prints a warning and
            returns ``None``).
        file_name (str): if non-empty, the SVG is also rasterised/vectorised
            to this path (``*.pdf`` -> PDF via cairosvg, anything else -> PNG).

    Returns:
        IPython.display.SVG or None: the SVG object for inline rendering, or
        ``None`` when the legends check fails.

    NOTE (known issue, left unchanged)
        The intermediate file is the hardcoded name ``'d2d.svg'`` in the *current
        working directory*, then deleted. Two concurrent calls would clobber
        each other's file.
    """
    if legends and len(legends) < len(mols):
      print('legends is too short')
      return None

    mols_per_row = min(len(mols), mols_per_row)
    # number of grid rows; ceil so the last partial row still fits
    rows = math.ceil(len(mols)/mols_per_row)
    # MolDraw2DSVG(width, height, panelWidth, panelHeight)
    d2d = rdMolDraw2D.MolDraw2DSVG(mols_per_row*size,rows*size,size,size)
    d2d.drawOptions().minFontSize = min_font_size
    if legends:
       d2d.DrawMolecules(mols, legends=legends)
    else:
        d2d.DrawMolecules(mols)
    d2d.FinishDrawing()

    if file_name:
      with open('d2d.svg', 'w') as f:
        f.write(d2d.GetDrawingText())
        if 'pdf' in file_name:
           cairosvg.svg2pdf(url='d2d.svg', write_to=file_name)
        else:
           cairosvg.svg2png(url='d2d.svg', write_to=file_name)
        os.remove('d2d.svg')

    return SVG(d2d.GetDrawingText())


def plot_molecules_from_list(smiles_list, max_num, img_size=(150, 150), imgs_per_row=5):
    """Render up to ``max_num`` SMILES as a matplotlib grid and return it as HTML.

    Args:
        smiles_list (List[str]): SMILES strings; the literal string ``"NAN"``
            is skipped (the pandas "not a number" marker leaking in from CSVs).
        max_num (int): cap on the number of molecules shown.
        img_size (Tuple[int, int]): per-cell pixel size used both for the RDKit
            raster and for the figure sizing (100 px = 1 inch).
        imgs_per_row (int): grid width.

    Returns:
        str: an ``<img src='data:image/png;base64,...'>`` HTML snippet of the
        rendered grid (handy for embedding in HTML reports).

    Note:
        ``plt.show()`` is called before ``savefig``; since the figure is not
        closed in between, the saved PNG is the same grid that was displayed.
    """
    # NOTE: invalid SMILES -> ``None`` here; ``Draw.MolToImage(None, ...)``
    #       raises, so a malformed string makes the whole grid fail.
    molecules = [Chem.MolFromSmiles(smile) for smile in smiles_list if smile != "NAN"]
    molecules = molecules[:max_num]
    # ceil division written as ``q + (r > 0)``
    rows = len(molecules) // imgs_per_row + (len(molecules) % imgs_per_row > 0)

    # Calculate figure size: width is fixed, height depends on the number of rows
    fig_width = imgs_per_row * (img_size[0] / 100)  # Convert pixels to inches
    fig_height = rows * (img_size[1] / 100)  # Convert pixels to inches
    fig, axs = plt.subplots(rows, imgs_per_row, figsize=(fig_width, fig_height))
    
    # ``flatten`` normalises the 1-D/2-D axes array so indexing is uniform
    # (matplotlib returns a bare Axes, not an array, when rows == cols == 1).
    axs = axs.flatten()
    # turn off the unused trailing cells
    for i in range(len(molecules), len(axs)):
        axs[i].axis('off')
    for i, mol in enumerate(molecules):
        img = Draw.MolToImage(mol, size=img_size)
        axs[i].imshow(img)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Convert plot to HTML image
    tmpfile = BytesIO()
    plt.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)
    return html

    
def plot_smiles_from_df(data_sorted,column_name):
    """ As the name suggests
    if ID column is given it will be printed with every batch of 9
    if ID is added it also shows the molecular weight to the compound in the name

    NOTE (known issue, left unchanged)
        无法运行：``hf`` 未导入/未定义（``NameError``），``display`` 也未定义
        （应为 ``IPython.display.display``）。保留原样仅作参考。
    """
    list_smiles = []
    sample_id_list = []
    for idx, i in enumerate(data_sorted[column_name]):
        # ``hf`` stands for a helper module (``helper_functions``) that was
        # never imported here -> NameError.
        weight = hf.calculate_weight(i)
        string = f"{column_name}_{idx}_{weight:.2f}"
        sample_id_list.append(string)
        list_smiles.append(Chem.MolFromSmiles(i))
        # flush every 9 molecules (a 3x3 page of depictions)
        if len(list_smiles)==9 or idx == len(data_sorted)-1:
            pic = Draw.MolsToGridImage((list_smiles), subImgSize=(250,250), legends = sample_id_list)
            display(pic)
            list_smiles = []
            sample_id_list = []



def plot_compare_scatter_plot(df_orig, df_sim, name="Plot", transp=0.50, style=["sim","orig","both"], direction=False):
    """Plots scatter plot from two dataframes on top of each other. The direction ticker can be selected to 
    distinguish between positive and negative intensities of the spectra (then displayed in 4 colours
    TODO: the sim or orig alone plotting is not working yet

    NOTE (known issue, left unchanged)
        1. ``style`` 默认值是 **list**，但判断写成 ``style == "orig"`` /
           ``style == "both"`` → 恒为 False，所以散点从来不会被画出来，
           只会显示一个带 legend 的空坐标系。
        2. ``pd`` 在本文件中从未导入 → ``pd.concat`` 处 ``NameError``。
        3. ``df_orig`` / ``df_sim`` 被原地写入 ``Phase`` 列（副作用）。
        4. ``direction=False`` 时 ``np.where`` 的两个分支完全相同，
           ``-ve_orig`` / ``-ve_sim`` 永远不会出现。
    """

    scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
    # intensity = np.array(df_orig['Intensity'].astype(float))
    # Tag each point with a phase label used later for per-group colouring.
    if direction:
        df_orig['Phase'] = np.where(df_orig['Intensity']>=0, '+ve_orig', '-ve_orig')
    else:
        # both branches identical -> everything is '+ve_orig'
        df_orig['Phase'] = np.where(df_orig['Intensity']>=0, '+ve_orig', '+ve_orig')
   

    scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
    # Simulated peaks carry a signed ``direction`` column instead of Intensity.
    if direction:
        df_sim['Phase'] = np.where(df_sim['direction']>=0, '+ve_sim', '-ve_sim')
    else:
        df_sim['Phase'] = np.where(df_sim['direction']>=0, '+ve_sim', '+ve_sim')

     
    
    # Concatenate coordinates in the same order as the concatenated Phase frame
    # below, so positional indices stay aligned.
    scatter_x = np.array(scatter_x_1 + scatter_x_2 )             
    scatter_y = np.array(scatter_y_1 + scatter_y_2 ) 
    df_orig = df_orig[['F2 (ppm)', 'F1 (ppm)', "Phase"]]
    df_sim = df_sim[['F2 (ppm)', 'F1 (ppm)', "Phase"]]

    df = pd.concat([df_orig, df_sim], axis=0) 
    group = df['Phase']  

    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    # NOTE: see issue 1 above - these comparisons never succeed for the default
    # list value, so no scatter is drawn.
    if (style == "orig") or (style == "both"):
        for g in np.unique(group):
            i = np.where(group == g)
            ax.scatter(scatter_x[i], scatter_y[i], label=g,  alpha=transp)

    ax.legend()
    plt.title(name)
    # 2-D NMR axes: 1H/13C ppm ranges, 0-11 (F2) and 0-200 (F1)
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    # NMR convention: origin at the top-right corner
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()



    
def plot_compare_scatter_plot_without_direction(df_orig, df_sim, name="Plot", transp=0.50):
    """Plots scatter plot from two dataframes on top of each other.

    Blue = real spectrum, red = simulated. Same 2-D NMR axis convention as
    :func:`plot_compare_scatter_plot`.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label="Real",  alpha=transp, color="blue")

    scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label="Simulated",  alpha=transp, color="red")

    ax.legend()
    plt.title(name)
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    
def plot_compare_scatter_plot_without_direction_all(df_real, df_acd, df_mnova, df_dft, df_ml, labels, transp=0.50):
    """Plots scatter plot from all simulation techniques on top of each other.

    Args:
        df_real: experimental HSQC/COSY peaks.
        df_acd: ACD/Labs prediction.
        df_mnova: Mnova prediction.
        df_dft: DFT-derived prediction.
        df_ml: MMST (machine-learning) prediction.
        labels (Sequence[str]): 5 legend labels, in the order
            ``[real, acd, mnova, dft, ml]``.
        transp (float): alpha used for the four simulated series; the real
            peaks stay opaque (alpha=1, green squares) as the reference.

    Returns:
        None (draws with ``plt.show``).
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    # Reference series: opaque green squares so it reads through the overlays.
    scatter_x_1 = list(np.array(df_real['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_real['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label=labels[0],  alpha=1, color="green", s=40, marker="s")

    scatter_x_2 = list(np.array(df_acd['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_acd['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[1],  alpha=transp, color="blue")
    
    scatter_x_2 = list(np.array(df_mnova['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_mnova['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[2],  alpha=transp, color="red")
    
    scatter_x_2 = list(np.array(df_dft['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_dft['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[3],  alpha=transp, color="orange")
    
    scatter_x_2 = list(np.array(df_ml['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_ml['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[4],  alpha=transp, color="m")
    
    ax.legend()
    plt.title("All spectra together")
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()
