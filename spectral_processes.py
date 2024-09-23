from lmfit.models import SplineModel, LorentzianModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob,os

rate = 78139901249.54921
rate *= 1e-9

def load( n = 6):
    if n == 6:
        folder_path = 'spectroscopy_data/peaks_6'
    if n == 5:
        folder_path = 'spectroscopy_data/peaks_5'
    csv_files = glob.glob(os.path.join(folder_path, '*.CSV')) 
    loaded_datasets = []
    i=0
    for file in csv_files:
        i +=1
        df = pd.read_csv(file)
        loaded_datasets.append(df)
        print(f"{i} Loaded {file}:")  # Display first few rows of each file
    return loaded_datasets


def get_dataset(n, data_array):
    x = data_array[n]['[s]'] * 1000
    y = data_array[n]['CH1[V]'] * 100
    return x,y

def move_ave_proced(arr, window_size=10):
    
    window_size = 10
    numbers_series = pd.Series(arr)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1:]
    return final_list

def move_ave(x,y,window_size = 10):
    x = np.array(move_ave_proced(x, window_size=window_size))
    y = np.array(move_ave_proced(y, window_size=window_size))
    return x,y

def get_knots(npoints = 10 , dataset=None, x=None, y=None, plotting = False):
    '''
        gets the knots positions at local minima in constants steps
        if plotting True : returns tuple xknots, yknots
    '''
    if dataset != None:
        x = dataset['[s]']
        y = dataset['CH1[V]']
    step = int(len(x)/npoints)
    knots = np.zeros(npoints, dtype=int)
    cutleft = 0
    cutright = step
    for n in range(npoints):
        localmin = y[cutleft:cutright].argmin()
        index_localmin = int(cutleft + localmin)
        knots[n] = index_localmin
        cutleft += step
        cutright += step
    if plotting:
        print('the knots are at second indices:', knots)
        print('the knots are at yvals:', y[knots])
        return (knots, y[knots])
    else:
        print('the knots are at seconds indices:', knots)
        return knots


def spline_model(xknots_ind, x,y, plot = False):
    bkg = SplineModel(prefix='bkg_', xknots=x[xknots_ind])
    params = bkg.guess(y  , x)
    for key in params.keys():
        # print(params[key])
        params[key].min = params[key].value - 0.2
        params[key].max = params[key].value + 0.2
    model =   bkg

    # init = model.eval(params, x=x) #for evaling initial guess
    out = model.fit(y, params, x=x) 
    print(out.fit_report(min_correl=0.3))
    knot_yvals = np.array([o.value for o in out.params.values() if o.name.startswith('bkg')])
    comps = out.eval_components()
    if plot:
        plt.scatter(x, y, s=1, label = 'og data')
        plt.plot(x, out.best_fit, label='best fit bkg', color= 'red')
        plt.plot(x[xknots_ind], knot_yvals, 'o', color='black', label='spline knots values')
        plt.legend()
        plt.grid()
        plt.title('Spline Model Fit')
        plt.show()
    return comps['bkg_']

def subtract_bkg(bkg, y_old, x, plot_orientacne = False, left =None, right=None):
    y =( y_old - bkg)*100
    if plot_orientacne:
        plt.scatter(x, y, s = 0.1, label = 'y_new')
        plt.scatter(x, y_old*100+300, s=0.1, label = 'y_old', color= 'red')
        plt.legend()
        plt.xlim(left, right)
        plt.minorticks_on()
        plt.grid(which='minor', alpha =0.1)
        plt.grid(which='major')

        plt.show()
    return y

def model_peaks(peak_num, centers, x, y , plot=False, amplitudes = None):
    if np.array(amplitudes).any() == None:
        amplitudes = [80]*6
    lorentzians = np.zeros(peak_num, dtype=object)
    for i in range(peak_num):
        lorentzians[i] = LorentzianModel(prefix=('l'+f'{i}'+'_'))
        if i == 0:
            params = lorentzians[i].make_params(
                center = dict(value = centers[i], min = centers[i] -0.2, max = centers[i]+0.2),
                sigma = 0.2,
                amplitude = dict(value = amplitudes[i], min = 0)
            )
        else:
            params.update(lorentzians[i].make_params(
                center = dict(value = centers[i], min = centers[i] -0.2, max = centers[i]+0.2),
                sigma = dict(value = 0.2, max = 1),
                amplitude = 80
            ))

    model = lorentzians[0]
    for i in range(1,6):
        model += lorentzians[i]

    init = model.eval(params, x=x)
    out = model.fit(y, params, x=x)
    print(out.fit_report(min_correl=0.3))
    comps = out.eval_components()
    if plot:
        plt.plot(x, comps['l0_'], label='l0')
        plt.plot(x, comps['l1_'], label='l1')
        plt.plot(x, comps['l2_'], label='l2')
        plt.plot(x, comps['l3_'], label='l3')
        plt.plot(x, comps['l4_'], label='l4')
        plt.plot(x, comps['l5_'], label='l5')

        plt.plot(x, out.best_fit, label='best fit', color= 'black', alpha = 1)
        plt.scatter(x,y,s=0.1, alpha = 0.5, label = 'og')
        plt.axhline(0, alpha = 0.3, color ='red')
        # plt.xlim(950,956)
        plt.legend()
        plt.show()
    return out

def eval_peaks(out, group = 3):
    result_peaks = np.zeros(6)
    result_stdrr = np.zeros(6)
    i = 0
    for name, param in out.params.items():
        if 'center' in name:
            print(f'{name:7s} {param.value:11.5f} {param.stderr:11.5f}')
            result_peaks[i] = param.value
            result_stdrr[i] = param.stderr
            i+=1 
    print(result_peaks)

    transitionsf3 = [0, 151.21, 352.45000000000005] + [ 75.605 ,251.83 , 176.225]
    transitionsf4 = [0, 201.24, 452.24] + [100.62 ,326.74 ,226.12]
    result_peaks -= result_peaks[0]
    if group == 3:
        plt.bar(result_peaks*rate, height=np.ones(6), width=3, color = 'red')
        plt.bar(transitionsf3, height=np.ones(6), width=1, color = 'black', linestyle ='-')
        plt.title('F3 transitions')
        plt.show()
    elif group == 4:
        plt.title('F4 group')
        plt.bar(result_peaks*rate, height=np.ones(6), width=3, color ='green')
        plt.bar(transitionsf4, height=np.ones(6), width=1, color = 'blue')
        plt.show()
    return np.array([result_peaks, result_stdrr]).T

def cut_data(x ,y , left, right):
    left = np.abs(x-left).argmin()
    right = np.abs(x-right).argmin()
    return x[left:right], y[left:right]

