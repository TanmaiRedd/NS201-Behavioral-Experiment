import os
import time
from pysinewave import SineWave
import numpy as np
import matplotlib.pyplot as plt
import random

def singleTrial(pitch1, pitch2, sinewave):
    # first plot

    sinewave.set_pitch(pitch1)
    # print(sinewave)
    # Turn the sine wave on.
    '''
    Fs = 8000
    sample = 8000
    x = np.arange(sample)
    y = np.sin(2 * math.pi * freq1 * x / Fs)
    plt.plot(x, y)
    sinewave.play()
    plt.show()
    # Sleep for 4 seconds, as the sinewave keeps playing.
    '''
    sinewave.play()
    time.sleep(1)
    sinewave.stop()
    time.sleep(1)
    '''
    # second plot
    sinewave = SineWave(pitch=pitch2)
    Fs = 8000
    sample = 8000
    x = np.arange(sample)
    y = np.sin(2 * math.pi * freq2 * x / Fs)
    plt.plot(x, y)
    '''
    sinewave.set_pitch(pitch2)
    sinewave.play()
    #plt.show()
    time.sleep(1)
    sinewave.stop()
    time.sleep(2)


def main():
    print('Hi! Thanks for taking part in this experiment. Please wear your headphones.')
    time.sleep(3)
    print('For this task, you will be listening to two tones. Please indicate if you think they are of the same frequency or not.')
    time.sleep(3)
    print('Please press 1 if you think they are the same frequency, and 0 if you think they are different.')
    time.sleep(3)
    name = input('Enter participant\'s name: ')
    time.sleep(5)

    sinewave = SineWave()
    if not os.path.exists('NS201_ParticipantData'):
        os.mkdir('NS201_ParticipantData')
    filename = 'NS201_ParticipantData/participant' + name + '.txt'
    file_object = open(filename, "w+")
    file_object.write('Participant: ' + name + '\n')
    #trial run: also tests their minimum distinguishable frequency
    freq1s = np.array([1000, 500])

    freq1s = np.random.permutation(freq1s)
    for freq1 in freq1s:
        trialpitch1 = (2 * np.log2(freq1 / 440)) - 9
        sinewave.set_pitch(trialpitch1)
        print('Ignore this sound!')
        sinewave.play()
        time.sleep(3)
        sinewave.stop()
        time.sleep(3)
        print('TRIAL RUN')
        file_object.write('Frequency measured ' + str(freq1) + '\n')


        minFreqq = 40
        if freq1 == 1000:
            minFreq = 60
        elif freq1 == 500:
            minFreq = 40
        else:
            minFreq = 20

        trialFreq2 = np.array([freq1, freq1 + minFreq, freq1 + minFreq*2, freq1 + minFreq*3, freq1 + minFreq*4])
        Trialresponse = np.zeros(trialFreq2.size)
        trialpitch2 = (2 * np.log2(trialFreq2 / 440)) - 9

        trialpitch2 = np.random.permutation(trialpitch2)
        for i in range(trialFreq2.size):
            singleTrial(trialpitch1, trialpitch2[i], sinewave)
            inputx = int(input('Were the sounds of the same frequency? Enter 1 for yes and 0 for no '))
            while not(inputx == 1 or inputx == 0):
                inputx = int(input('Enter a valid response. Enter 1 for yes and 0 for no '))

            Trialresponse[i] = inputx

            if trialpitch2[i] == trialpitch1:
                print('correct response is 1')
            else:
                print('correct response is 0')
            time.sleep(1)
            '''
            if Trialresponse[i] == 0 and i != 0:
                minFreqq = trialFreq2[i] - freq1
            '''


        file_object.write('Minimum distinguishable frequency: ' + str(minFreq) + '\n')
        time.sleep(8)

        print('Experiment begins now')
        time.sleep(3)
        frequency_range1 = np.array(range(freq1-(5*(minFreq-10)), freq1+(6*(minFreq-10)), minFreq-10))
        frequency_range = np.zeros(frequency_range1.size-1)
        #frequency_range1[int(frequency_range.size/2)] = frequency_range[int(frequency_range.size/2)]+(minFreq-10)
        for i in range(int(frequency_range1.size)-1):
            if i<int(frequency_range.size/2):
                frequency_range[i] = frequency_range1[i]
            else:
                frequency_range[i] = frequency_range1[i+1]

        #file_object.write('Frequency range: ' + str(frequency_range) + '\n')
        #np.choice()1
        #print(frequency_range)
        no_of_trials = frequency_range.size*3
        trials = np.zeros(no_of_trials)
        random.seed(0)
        index = np.zeros(frequency_range.size*3)
        for i in range(frequency_range.size*2):
            index[i] = int(random.randint(0, np.size(frequency_range)-1))
            trials[i] = frequency_range[int(index[i])]

        trials[frequency_range.size*2:frequency_range.size*3] = frequency_range
        file_object.write('Trials: ' + str(trials) + '\n')

        #print(trials)
        trials = np.random.permutation(trials)
        #print(trials)
        no_of_triplicates = np.zeros(frequency_range.size)
        for j in range(frequency_range.size):
            for i in range(no_of_trials):
                if trials[i] == frequency_range[j]:
                    no_of_triplicates[j] = no_of_triplicates[j] + 1

        file_object.write('Number of triplicates: ' + str(no_of_triplicates) + '\n')
        #print(no_of_triplicates)
        #a = 5
        #b = 8
        # frequency = 440 * 2 ^ ((pitch - 9) / 2)
        #f1 = [a, a, b, b, a, a, a, b, a, b]
        #f2 = [b, a, a, b, b, a, a, a, b, a]
        #print(trials)
        pitch1 = (2 * np.log2(freq1 / 440)) - 9
        pitch2 = (2 * np.log2((trials+1) / 440)) - 9
        #print(pitch1)
        #print(pitch2)
        response = np.zeros(no_of_trials)
        CorrectOut = np.zeros(no_of_trials)
        file_object.write('Responses & correct answer: '+'\n')
        for j in range(no_of_trials):
            #j = trials[i]
            #print(j)
            stim = np.random.permutation([pitch1, pitch2[j]])
            singleTrial(stim[0], stim[1], sinewave)
            CorrectOut[j] = int(pitch1 == pitch2[j])
            inputx = int(input('Trial'+str(j) + ': Were the sounds of the same frequency? Enter 1 for yes and 0 for no '))
            while not(inputx == 1 or inputx ==0):
                inputx = int(input('Enter a valid response. Enter 1 for yes and 0 for no and then press enter '))
            response[j] = inputx
            file_object.write(str(response[j]) +" " + str(CorrectOut[j])+ '\n')
            # time.sleep(2)

        # response = np.random.randint(0, 2, no_of_trials)
        percent_correctDetections = np.zeros(frequency_range.size)
        '''
        for i in range (no_of_trials):
            if trials[i] == freq1:
                CorrectOut[i] = 1
        '''
        for j in range(frequency_range.size):
            for i in range(no_of_trials):
                if frequency_range[j] == trials[i]:
                    if response[i] == CorrectOut[i]:
                        percent_correctDetections[j] = percent_correctDetections[j] + 1/no_of_triplicates[j]

        file_object.write('percent correct: ' + str(percent_correctDetections) + '\n')
        file_object.write('\n')
        file_object.write('\n')
        #print(percent_correctDetections)
        #plt.plot((frequency_range-freq1), percent_correctDetections)
        freq_diff = frequency_range - freq1
        abspercentcorrect = np.zeros(int(frequency_range.size / 2))
        for i in range(int(freq_diff.size)):
                #abspercentcorrect[i-1] = percent_correctDetections[i]/no_of_triplicates[i]
            if i<int(freq_diff.size/2):
                abspercentcorrect[i] = percent_correctDetections[i]/2 + percent_correctDetections[freq_diff.size - i - 1]/2
            else:
                break

        np.savez('NS201_ParticipantData/Participant: ' + name+str(freq1)+'.npz', response, CorrectOut, pitch1, pitch2)

        file_object.write('Absolute percent correct: '+str(abspercentcorrect)+'\n')
        plt.plot(freq_diff[int(freq_diff.size / 2):], np.flip(abspercentcorrect))
        plt.title('Participant: ' + name+", Frequency: "+str(freq1))
        plt.xlabel('Frequency difference (Hz)')
        plt.ylabel('Percent correct')
        plt.savefig('participant_' + name + '_frequency_' + str(freq1) + '.png')
        plt.show()

        print('End of sub-experiment')
        time.sleep(3)
    file_object.close()
    print('End of experiment')

main()