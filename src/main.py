# -*- coding: utf-8 -*-

import time
import numpy as np
from datetime import datetime

import matplotlib
from matplotlib import pyplot as plt
# Disable showing figures to prevent random GPU failures.
matplotlib.use('Agg')

import os
import random as python_random
import codecs

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, plot_precision_recall_curve
from sklearn.utils import class_weight

import tensorflow as tf
from keras.callbacks import EarlyStopping

from models import deep_cnn_sequential
from dataset import DatasetGenerator

#==============================================================================
# Helper classes and methods
#==============================================================================
def set_fixed_seed():
    """
    Set fixed random values and disable GPU to enable fixed training.
    """
    # Seed value
    # Apparently you may use different seed values at each stage
    seed_value = SEED_VALUE

    # 0. Disable GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set `python` built-in pseudo-random generator at a fixed value
    python_random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


# Hyperbolic tangent function
# Custom scheduler obtained from:
# https://medium.com/@abhismatrix/neural-network-model-training-tricks-61254a2a1f6b
class CustomScheduleTanh(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps=4000, phase_step = 3000, max_lr = .001):
        super(CustomScheduleTanh, self).__init__()
        self.phase_step = phase_step
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.lr = 0
        self.step = 0
        
    def __call__(self, step):
        self.step=step
        current_shifted_step = tf.math.minimum(tf.math.maximum((step-self.warmup_steps),-3.0)*5/(self.phase_step-self.warmup_steps),5.)
        arg1 = -tf.math.tanh(current_shifted_step-2.) + tf.constant(1.)
        arg2 = self.max_lr*step/(self.warmup_steps)
        arg3 = tf.math.maximum(self.max_lr*arg1/2.,self.max_lr/10000)
 
        lr = tf.math.minimum(arg2, arg3)
        self.lr = lr
        return lr


#==============================================================================
# Hyperparameters
#==============================================================================
DIR = 'path/to/ENST/data'
LABELS = ['no', 'yes'] # Has onset: no = 0, yes = 1
DRUM_INSTRUMENTS = ['bd', 'sd']
NUM_CLASSES = len(LABELS)
PLOT_SAVE_LOCATION = ''

CHANNELS = [2048]  # [1024, 2048, 4096]
MEL_BANDS = 80
TIME_FRAMES = 12
DIFF_FROM_ONSET_MS = 0.03
THRESHOLD_FREQ = 15000

BATCHES = [64, 256, 512]
EPOCHS = 175
PATIENCE = 150
TRAIN_TEST_SPLIT = 0.10
TRAIN_VAL_SPLIT = 0.20
TRAINING_DATA_SPLIT_BY = 'WINDOWS'  # 'SONGS'
# Categorical cross-entropy expects labels to be provided in a one-hot representation (0, 1).
LOSS_FUNCTION = 'categorical_crossentropy'
LEARNING_RATE = 0.001

INPUT_SHAPE = (MEL_BANDS, TIME_FRAMES, len(CHANNELS))
PRED_LAYER_ACTIVATION = 'sigmoid'
METRICS = ['acc']
PREC_REC_FSCORE_AVERAGE = 'macro'  # None # 'weighted' # 'micro'

FIXED_SEED = False
SEED_VALUE = 0
if FIXED_SEED:
    set_fixed_seed()

#==============================================================================
# Prepare data      
#==============================================================================

N = 8  # How many outer loops. Each contributes to the mean and standard deviation.
M = 4  # Find the best (minimum) validation loss and fscore among M runs.


def prepare_data(drum_instrument):
    # Set to global scope for easy access in other functions.
    global DRUM_INSTRUMENT
    global TRAIN_DATA
    global TRAIN_LABELS
    global TRAIN_LABELS_1D
    global TEST_DATA
    global TEST_LABELS

    DRUM_INSTRUMENT = drum_instrument

    dsGen = DatasetGenerator(label_set=LABELS,
                             sample_rate=44100,
                             channels=CHANNELS,
                             mel_bands=MEL_BANDS,
                             time_frames=TIME_FRAMES,
                             diff_from_onset_ms=DIFF_FROM_ONSET_MS,
                             threshold_freq=THRESHOLD_FREQ,
                             drum_instrument=drum_instrument)

    # Load DataFrame with paths/labels for training and validation data.
    dsGen.load_datafiles(DIR)

    # Split data either by songs or windows. Produces different results.
    if TRAINING_DATA_SPLIT_BY == 'SONGS':
        dsGen.apply_train_test_split(test_size=TRAIN_TEST_SPLIT, random_state=SEED_VALUE)
        TRAIN_DATA, TRAIN_LABELS = dsGen.get_data(mode='train')
        TEST_DATA, TEST_LABELS = dsGen.get_data(mode='test')
    else:
        TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = dsGen.apply_train_test_split_by_windows(test_size=TRAIN_TEST_SPLIT, shuffle_train_data=True)

    # Needed for class weights.
    TRAIN_LABELS_1D = np.argmax(TRAIN_LABELS, axis=1)
    print('Training data size: ', len(TRAIN_DATA))
    print('Test data size: ', len(TEST_DATA))
    print('Marked onsets count in training labels: ', len(list(filter(lambda x: x == 1, TRAIN_LABELS_1D))), '/', len(TRAIN_LABELS_1D))
    print('Marked onsets count in test labels: ', len(list(filter(lambda x: x == 1, TEST_LABELS))), '/', len(TEST_LABELS))


#==============================================================================
# Run
#==============================================================================

def main():
    table_data = {}
    # Train CNN for each drum instrument.
    for drum_instrument in DRUM_INSTRUMENTS:
        prepare_data(drum_instrument)
        table_data[drum_instrument] = {}

        # Loop through different batch sizes.
        for batch_size in BATCHES:
            precisions = []
            recalls = []
            fscores = []
            min_val_losses = []

            # Evaluation framework.
            for n in range(N):
                best_precision = float('-inf')
                best_recall = float('-inf')
                best_fscore = float('-inf')
                best_min_val_loss = float('inf')

                # The learning algorithm.
                # Choosing the best run based on training results among M runs.
                for m in range(M):
                    min_val_loss, precision, recall, fscore, acc, val_acc, loss, val_loss, id = run(batch_size)

                    # Pick the best run based on the minimum validation loss.
                    if min_val_loss < best_min_val_loss:
                        best_min_val_loss = min_val_loss                
                        best_fscore = fscore
                        best_precision = precision
                        best_recall = recall

                    plot(acc, val_acc, 'Accuracy', 'Validation accuracy', PLOT_SAVE_LOCATION, id, batch_size)
                    plot(loss, val_loss, 'Loss', 'Validation  loss', PLOT_SAVE_LOCATION, id, batch_size)

                precisions.append(best_precision)
                recalls.append(best_recall)
                fscores.append(best_fscore)
                min_val_losses.append(best_min_val_loss)

            # Get results for LaTeX table.
            p_mean = np.mean(precisions)
            r_mean = np.mean(recalls)
            f_mean = np.mean(fscores)
            min_val_loss_mean = np.mean(min_val_losses)

            p_std = np.std(precisions)
            r_std = np.std(recalls)
            f_std = np.std(fscores)
            min_val_loss_std = np.std(min_val_losses)

            print(p_mean)
            print(r_mean)
            print(f_mean)
            print(min_val_loss_mean)

            print(p_std)
            print(r_std)
            print(f_std)
            print(min_val_loss_std)

            table_data[drum_instrument][batch_size] = {
                'p_mean': p_mean,
                'p_std': p_std,
                'r_mean': r_mean,
                'r_std': r_std,
                'f_mean': f_mean,
                'f_std': f_std
            }

    # Enable automatic LaTeX table creation of the results.
    # create_latex_table(table_data, id)

    
def run(batch_size):
    """
    One run. Build the model, train and evaluate.
    """
    start = time.time()
    
    model = get_model()
    history = train(model, batch_size)

    y_true, y_pred = predict(model)
    precision, recall, fscore, acc_score = get_metrics(y_true, y_pred)
    tn, fp, fn, tp = get_confusion_matrix(y_true, y_pred)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Write your own logging.
    # log(history, start, precision, recall, fscore, acc_score, tn, fp, fn, tp)
    min_val_loss = min(history.history['val_loss'])

    now = datetime.now()
    id = now.strftime('%Y%m%d%H%M%S')
    elapsed_s = time.time() - start
    elapsed = time.strftime('%H:%M:%S', time.gmtime(elapsed_s))

    print(elapsed)

    print('Accuracy: ', acc_score)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F-score: ', fscore)

    print('TN: ', tn)
    print('FP: ', fp)
    print('FN: ', fn)
    print('TP: ', tp)

    return min_val_loss, precision, recall, fscore, acc, val_acc, loss, val_loss, id


def get_model():
    """
    Build and compile a CNN model. 
    """
    # Reset scheduled learning rate.
    learning_rate_schedule = CustomScheduleTanh(warmup_steps=3000, phase_step=25000, max_lr=LEARNING_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    model = deep_cnn_sequential(INPUT_SHAPE, NUM_CLASSES, act=PRED_LAYER_ACTIVATION)
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION, metrics=METRICS)
    return model


def train(model, batch_size):
    """
    Train the model and return history results.
    """
    global TRAIN_LABELS_1D
    global TRAIN_DATA
    global TRAIN_LABELS

    callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.01, patience=PATIENCE, verbose=1, mode='auto')]
    # Balance imbalanced onset classes.
    class_weights = class_weight.compute_class_weight('balanced', np.unique(TRAIN_LABELS_1D), TRAIN_LABELS_1D)

    history = model.fit(x=TRAIN_DATA, 
                        y=TRAIN_LABELS, 
                        batch_size=batch_size,
                        epochs=EPOCHS,
                        verbose=1,
                        callbacks=callbacks,
                        validation_split=TRAIN_VAL_SPLIT,
                        class_weight=class_weights)
    return history


def predict(model):
    global TEST_DATA
    global TEST_LABELS

    y_true = TEST_LABELS
    y_pred = model.predict_classes(x=TEST_DATA, verbose=1)
    return y_true, y_pred

    
def get_metrics(y_true, y_pred):
    acc_score = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=PREC_REC_FSCORE_AVERAGE)
    return precision, recall, fscore, acc_score


def get_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def plot(metric1, metric2, label1, label2, save_location, id, batch_size):
    """
    Creates and saves the plotted figure.
    """
    try: 
        fig = plt.figure()
        plt.plot(metric1, label=label1)
        plt.plot(metric2, label=label2, linestyle='dashed')
        plt.legend()
        plt.xlabel('Epoch')
        plt.grid(linestyle='dotted')
        # plt.ylim(top=)
        # plt.show()
        plt.savefig(save_location + id + '_' + DRUM_INSTRUMENT + '_' + str(EPOCHS) + '_' + str(batch_size) + '.pdf')
        plt.clf()
        plt.cla()
        plt.close(fig=fig)
    except Exception as e:
        print('Failed to create plot: ', e)


def create_latex_table(data, id):
    """
    Creates a LaTeX table with different batch sizes, drum instruments and input metrics.
    """
    bd = data['bd']
    sd = data['sd']
    
    filename = 'LatestResults.tex'
    file = r'..\latex\tables\\' + filename

    if os.path.exists(file):
        f_temp = os.path.splitext(file)[0] # without extension
        os.rename(file, f_temp + '_' + id + '.tex')

    f = codecs.open(file, 'w', 'utf-8')
    
    f.write('\n' + r'\begin{table}' + '\n')
    f.write(r'  \centering' + '\n')
    f.write(r'  \caption{Results for each drum instrument with batch sizes 64, 256 and 512.}' + '\n')
    f.write(r'  \begin{tabular}{l c c c}' + '\n')
    f.write(r'  \textbf{Batch size} & Metric & BD & SD \\' + '\n')
    f.write(r'  \midrule' + '\n')
    f.write(r'  \midrule' + '\n')
    
    for batch_size in BATCHES:
        f.write('  ' + str(batch_size).rstrip('\n'))
        # 0.805 +- 0.02
        f.write(r'  & P & ' + r'$' + '{:.3}'.format(bd[batch_size]['p_mean']) + r' \pm ' + '{:.3f}'.format(bd[batch_size]['p_std']) + '$' + r' & ' + r'$' + '{:.3}'.format(sd[batch_size]['p_mean']) + r' \pm ' + '{:.3f}'.format(sd[batch_size]['p_std']) + '$' + r' \\' + '\n')
        f.write(r'  & R & ' + r'$' + '{:.3}'.format(bd[batch_size]['r_mean']) + r' \pm ' + '{:.3f}'.format(bd[batch_size]['r_std']) + '$' + r' & ' + r'$' + '{:.3}'.format(sd[batch_size]['r_mean']) + r' \pm ' + '{:.3f}'.format(sd[batch_size]['r_std']) + '$' + r' \\' + '\n')
        f.write(r'  & F & ' + r'$' + '{:.3}'.format(bd[batch_size]['f_mean']) + r' \pm ' + '{:.3f}'.format(bd[batch_size]['f_std']) + '$' + r' & ' + r'$' + '{:.3}'.format(sd[batch_size]['f_mean']) + r' \pm ' + '{:.3f}'.format(sd[batch_size]['f_std']) + '$' + r' \\' + '\n')
        # Don't write horizontal line on the last batch.
        if batch_size != BATCHES[-1]:
            f.write(r'  \midrule' + '\n')

    f.write(r'  \end{tabular}' + '\n')
    f.write(r'  \label{tab:ResultsTable}' + '\n')
    f.write(r'\end{table}' + '\n')
    f.close()


if __name__ == '__main__':
    main()

