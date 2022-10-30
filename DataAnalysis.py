# this code takes data from each .npz file of the 11 participants and performs statistical analysis on the same
# arrays in the .npz file:
# arr_0.npy: response
# arr_1.npy: CorrectOut
# arr_2.npy: pitch1 (single value)
# arr_3.npy: pitch2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import statistics
from scipy import stats

path = os.getcwd()

def openfile(filename):
    with np.load(filename) as data:
        pitch1 = data['arr_3.npy']
        pitch2 = data['arr_2.npy']
        response = data['arr_0.npy']
        # correctOut = data['arr_1.npy'] # not required since none of the frequencies match
        freq1 = (2**(pitch1+9)/2)*440
        freq2 = (2**(pitch2+9)/2)*440
    return freq1, freq2, response

def textfileExtract(filename, freq):
    f = open(filename, 'r',encoding='utf-8')
    lines = f.readlines()
    count = 0
    #print('check1')
    #print(len(lines))
    for ii in range(len(lines)):
        #print(i)
        linex = lines[ii]

        if linex == 'Frequency measured '+str(freq)+'\n':
            for i in range(ii,len(lines),1):
                line = lines[i]

                #print('check aaaa')
                abs_index=0
                triplicate_index=0
                percent_index=0
                words = line.split()
                triplicates = np.zeros([10])
                percentcorrect = np.zeros([10])
                absolutepercentcorrect = np.zeros([10])
                #print(words)
                #print(len(words))
                for j in range(len(words)):
                    word = words[j]
                    #print(word)
                    if word == 'Absolute':
                        abs_index = j
                        line_absolutepercentcorrect = line[abs_index:]
                        absolutepercentcorrect = extract(line_absolutepercentcorrect)
                        #print(absolutepercentcorrect)
                        count = count +1
                    if word == 'Number':
                        triplicate_index = j
                        line_triplicates = line[triplicate_index:]
                        triplicates = extract(line_triplicates)
                        #print(triplicates)
                        count = count + 1
                    if word == 'percent':
                        percent_index = j
                        line_edit = line+lines[i+1]
                        #print(line_edit)
                        line_percentcorrect = line_edit[percent_index:]
                        percentcorrect = extract(line_percentcorrect)
                        #print(percentcorrect)
                        count = count + 1
                    if count == 3:
                        return percentcorrect, triplicates, absolutepercentcorrect
                if count == 3:
                    break
        if count == 3:
            break
            #print('check2')




def extract(line):
    sub1 = '['
    sub2 = ']'
    idx1 = line.index(sub1)
    #print(idx1)
    idx2 = line.index(sub2)
    #print(idx2)
    arr = np.array([])
    stringnum = line[idx1+1:idx2]
    stringnum = stringnum.split()
    #print(stringnum)
    arr = np.array(stringnum, dtype=float)
    #print(arr)
    #print()
    #print('check3')
    #decimal_length = 0
    '''
    for idx in range(idx1, idx2):
        #res = res + line[idx]
        #print(idx)
        n = line[idx]
        if decimal_length != 0:
            decimal_length = decimal_length -1
            continue
        if n !=sub1 and n!=sub2 and n!='' and n!=' ' and n!= '  ':
            if n == '.':
                space_ind = 0
                for spacefind in range(idx , idx2):
                    c = line[spacefind]
                    if c == ' ':
                        space_ind = spacefind
                        #print(space_ind)
                        break
                decimal_length = space_ind - idx
                if decimal_length != 0:
                    #extract the decimal places
                    n = '0'+line[idx:space_ind]
                    #print(n)

                    arr[-1] = arr[-1] + float(n)
                #if idx
                else:
                    continue
                #idx = idx + decimal_length
    '''
    '''
                n1 = line[idx+1]
                n2 = line[idx+2]
                if n1 == ' ':
                    continue
                if n2 != ' ':
                    n = n1+n2
                else:
                    n = n1
                #print(idx)
                #print(idx1)
                arr[-1] = arr[-1] + 0.1*float(n)
                print(arr)
    '''
    '''
            else:
                if (n != '.'):
                    arr = np.append(arr, float(n))
                    #print(arr)
                else:
                    continue
            if arr.size == 10:
                break
    '''
    #print('check4')

    return arr


def removeAdditionalParticipants(array, ignore_participants):
    array_out = np.zeros((array.shape[0], array.shape[1]-ignore_participants.size, array.shape[-1]))
    c=0
    #print(array.shape[1])
    #print(ignore_participants.size)
    #print(array.shape[1]-ignore_participants.size)
    for i in range(array.shape[1]):
        #print(i)
        if (i+1) not in ignore_participants:
            array_out[:,c,:] = array[:,i,:]
            c = c+1
            #print(i)
    return array_out

def meancalc(array):
    # array is 3x11x5 array, we need output of 3x5 array (average across 2nd dim)
    arrayout = np.average(array, 1)
    #print(arrayout.shape)
    return arrayout


def plotter(array, arr_stdev):
    # array is 3x5 and so is the stdev

    freqs = [300, 500, 1000]
    freqdiff = [20, 40, 60]
    fig, ax = plt.subplots(3,1, figsize= (10,20))
    JND = np.zeros(array.shape[0])

    #print(array.shape)
    for i in range(array.shape[0]):
        xdata = (np.array(range(freqs[i]+((freqdiff[i]-10)), freqs[i]+(6*(freqdiff[i]-10)), freqdiff[i]-10)) - freqs[i])
        ydata = np.flip(array[i,:])
        #curvefit
        p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is an mandatory initial guess
        popt,pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
        x = np.linspace(0, xdata[-1], 5)
        y = sigmoid(x, *popt)
        maxdiff = y[1]-y[0]
        y_diff = y[1]+maxdiff
        index_maxdiff = (x[1]+x[0])/2
        for j in range(y.shape[0]-1):
            diff = y[j+1] - y[j]
            if (diff>maxdiff):
                maxdiff = diff
                y_diff = (y[j]+y[j+1])/2
                index_maxdiff = (x[j+1]+x[j])/2
        JND[i] = index_maxdiff
        #print()
        #print(index_maxdiff, maxdiff)
        ax[i].plot(index_maxdiff, y_diff, marker='o', label='JND')
        ax[i].plot(xdata, ydata, label='data')
        #ax[i].bar(xdata, ydata,yerr = arr_stdev[i,:])
        ax[i].plot(x,y, label='fit')
        ax[i].grid()
        ax[i].legend(loc='lower left')
        #plt.legend('data', 'fit')
        ax[i].set_ylabel('Percent correct')
        if i == 2:
            ax[i].set_xlabel('Frequency difference')

        ax[i].set_title('Frequency: '+str(freqs[i]))
    #plt.legend()
    plt.suptitle('Psychometric curves for frequency detection')

    plt.savefig(path+'/NS201_ParticipantData/PsychometricCurveAvg.png')
    plt.show()

    ## plot with error bars
    dist_stdev = np.zeros(array.shape[0])

    freqs = [300, 500, 1000]
    freqdiff = [20, 40, 60]
    fig, ax = plt.subplots(3, 1, figsize=(10, 20))
    for i in range(array.shape[0]):
        xdata = (np.array(
            range(freqs[i] + ((freqdiff[i] - 10)), freqs[i] + (6 * (freqdiff[i] - 10)), freqdiff[i] - 10)) - freqs[i])
        ydata = np.flip(array[i, :])
        # curvefit
        p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is an mandatory initial guess
        popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
        x = np.linspace(0, xdata[-1], 5)
        y = sigmoid(x, *popt)

        # calculate stdev of distribution
        dist_stdev[i] = statistics.pstdev(ydata)
        print('T-test for freq: ', freqs[i])
        print(stats.ttest_ind(y, ydata))

        maxdiff = y[1] - y[0]
        y_diff = y[1] + maxdiff
        index_maxdiff = (x[1] + x[0]) / 2
        for j in range(y.shape[0] - 1):
            diff = y[j + 1] - y[j]
            if (diff > maxdiff):
                maxdiff = diff
                y_diff = (y[j] + y[j + 1]) / 2
                index_maxdiff = (x[j + 1] + x[j]) / 2

        JND[i] = index_maxdiff
        # print()
        # print(index_maxdiff, maxdiff)
        ax[i].plot(index_maxdiff, y_diff, marker='o', label='JND')
        ax[i].errorbar(xdata, ydata, yerr=arr_stdev[i, :],label='data')
        #ax[i].bar(xdata, ydata )
        ax[i].plot(x, y, label='fit')
        ax[i].grid()
        ax[i].legend(loc='lower left')
        # plt.legend('data', 'fit')
        ax[i].set_ylabel('Percent correct')
        if i == 2:
            ax[i].set_xlabel('Frequency difference')
        ax[i].set_title('Frequency: ' + str(freqs[i]))
    plt.legend()
    plt.suptitle('Psychometric curves for frequency detection')
    plt.savefig(path+'/NS201_ParticipantData/PsychometricCurveAvgErrorBars.png')
    plt.show()
    return JND, dist_stdev

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def find_stddev(array):
    #return 3x5 array: std dev for 11 participants for each freq diff and each frequency
    array_out = np.zeros((array.shape[0], array.shape[-1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[-1]):
            array_out[i,j] = statistics.pstdev(array[i,:,j])

    standard_dev = np.mean(array_out, axis=1)
    print(standard_dev.shape)
    return array_out, standard_dev


def checkWebers(freqs, jnd):
    div = np.zeros(jnd.size)
    for i in range(div.size):
        div[i] = jnd[i]/freqs[i]
    plt.plot(freqs, div, label = 'data')
    x = np.array(range(300, 1000, 100))
    z = np.polyfit(freqs, div, 1)
    ang_coeff_div = z[0]
    intercept_div = z[1]
    f = np.poly1d(z)
    y = f(x)
    plt.plot(x, y, label = 'fit')
    plt.legend()
    plt.title('Weber\'s Law')
    plt.xlabel('Frequencies')
    plt.ylabel('JND/Frequency')
    plt.ylim([0, 0.5])
    plt.savefig(path+'/NS201_ParticipantData/Weberslawdiv.png')
    plt.show()

    # weber's law proper plot
    #popt = curve_fit(linear, jnd, freqs)
    x = np.array(range(20,200,40))
    #y = linear(x, *popt)
    z = np.polyfit(jnd, freqs,1)
    ang_coeff = z[0]
    intercept = z[1]
    f = np.poly1d(z)
    y = f(x)
    print()
    print('P-value for weber\'s law verification: ', stats.ttest_ind(y, freqs))
    print('p-value for aslope of weber\'s law: ', stats.ttest_ind(ang_coeff, 0))
    #print(stats.ttest_1samp(ang_coeff, 0))
    plt.plot(jnd, freqs, label = 'data')
    plt.plot(x, y, label = 'fit')
    plt.title('Weber\'s Law')
    plt.ylabel('Frequencies')
    plt.xlabel('JND/Frequency')
    plt.legend()
    plt.savefig(path+'/NS201_ParticipantData/WeberslawVerify.png')
    plt.show()

    return ang_coeff_div, intercept_div, ang_coeff, intercept


def linear(x,x_0, m, b):
    y = m(x-x_0) + b
    return y


def plot_persubject(array, ignore_participants, no_of_participants):
    participantname = np.zeros(no_of_participants-ignore_participants.size)
    xx=0
    for i in range(no_of_participants):
        if (i+1) not in ignore_participants:
            participantname[xx] = i+1
            xx=xx+1

    #print(participantname)

    fig, ax = plt.subplots(1,array.shape[0], figsize = (10,5))
    ax = ax.flatten()
    freqs = np.array([300, 500, 1000])
    freqdiff = np.array([20, 40, 60])
    for i in range(array.shape[0]):
        xdata = (np.array(range(freqs[i] + ((freqdiff[i] - 10)), freqs[i] + (6 * (freqdiff[i] - 10)), freqdiff[i] - 10)) - freqs[i])
        for j in range(array.shape[1]):
            ydata = np.flip(array[i, j,:])
            ax[i].plot(xdata, ydata)
            ax[i].set_title('Frequency: '+str(freqs[i]))
            ax[i].set_xlabel('Frequency difference')
            ax[i].set_ylabel('Percent correct')
    plt.savefig(path+'/NS201_ParticipantData/PsychometricEachSubject.png')
    plt.suptitle('Psychometric curves for frequency detection for all subjects')


    fig, ax = plt.subplots(array.shape[0], array.shape[1], figsize = (40,10))
    ax = ax.flatten()
    #print(ax.size)
    c=0
    for i in range(array.shape[0]):
        xdata = (np.array(range(freqs[i] + ((freqdiff[i] - 10)), freqs[i] + (6 * (freqdiff[i] - 10)), freqdiff[i] - 10)) - freqs[i])
        for j in range(array.shape[1]):
            ydata = np.flip(array[i, j,:])
            '''
            p0 = [max(ydata), np.median(xdata), 1, min(ydata)]  # this is an mandatory initial guess
            popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
            x = np.linspace(0, xdata[-1], 5)
            y = sigmoid(x, *popt)
            ax[c].plot(x, y, label='fit')
            '''
            ax[c].plot(xdata, ydata, label='data')
            if c<array.shape[1]:
                ax[c].set_title('Subject: '+str(participantname[c]))
            ax[c].set_xlabel('Frequency difference')
            ax[c].set_ylabel('Percent correct')
            ax[c].legend(loc='lower left')
            c=c+1
    #plt.legend()
    plt.savefig(path+'/NS201_ParticipantData/PsychometricEachSubject_diff.png')
    plt.suptitle('Psychometric curves for frequency detection for all subjects')
    plt.show()


def main():
    freqs = np.array([300, 500, 1000])
    num_participants = 11
    ignore_participants = np.array([1,2,4,7,9])
    path = os.getcwd()
    response = np.zeros((freqs.size, num_participants+1, 30)) #30 trials -> 30 responses
    percentcorrect = np.zeros((freqs.size, num_participants, 10))
    triplicates = np.zeros((freqs.size, num_participants, 10))
    absolutepercentcorrect = np.zeros((freqs.size, num_participants, 5))
    for freqnum in range(freqs.size):
        for subnum in range(1,num_participants+1,1):
            #if subnum != 1 and subnum !=2:
            if subnum not in ignore_participants:
                if subnum<=9:
                    numstr = '00'+str(subnum)
                else:
                    numstr = '0'+str(subnum)
                filename_txt = path+'/NS201_ParticipantData/NS201_ParticipantData/participant' + numstr+'.txt'
                numstr = numstr + str(freqs[freqnum])
                filename_npz = path+'/NS201_ParticipantData/Participant'+numstr+'.npz'
                freq1, freq2, response[freqnum, subnum, :] = openfile(filename_npz)
                #print(response[freqnum, subnum, :])
                percentcorrect[freqnum, subnum-1, :], triplicates[freqnum, subnum-1, :], absolutepercentcorrect[freqnum, subnum-1, :] = textfileExtract(filename_txt, freqs[freqnum])
            #print(subnum)
    #time.sleep(5)

    print()
    #remove bad participants
    absolutepercentcorrect_actual = removeAdditionalParticipants(absolutepercentcorrect, ignore_participants)
    #print(absolutepercentcorrect_actual.shape)
    # plotter for all participants i.e., 11x3 plots
    plot_persubject(absolutepercentcorrect_actual, ignore_participants, num_participants)


    # now we have to calculate mean
    avg_per_corr = meancalc(absolutepercentcorrect_actual)
    # calc stdev
    stdev_per_corr, standard_dev = find_stddev(absolutepercentcorrect_actual)
    print(standard_dev)
    # plotter : plots stuff and also calculates the JND
    jnd, dist_stdev = plotter(avg_per_corr, stdev_per_corr)
    print(jnd)

    # weber's law
    ang_coeff_div, intercept_div, ang_coeff, intercept = checkWebers(freqs, jnd)
    print(ang_coeff_div, intercept_div, ang_coeff, intercept)
            #print(subnum)
        #print()
        #print(freqs[freqnum])

main()


