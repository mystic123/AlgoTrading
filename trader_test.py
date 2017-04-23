import numpy as np
import pandas as pd
import csv
import os
import sys
import datetime as dt
import tensorflow as tf
from scipy.signal import argrelextrema

def _Accuracy(y, yHat):
    n = float(len(y))
    return np.sum(y == yHat) / n

#Create the MLP variables for TF graph
#_X: The input matrix
#_W: The weight matrices
#_B: The bias vectors
#_AF: The activation function
def _CreateMLP(_X, _W, _B, _AF):
    n = len(_W)
    for i in range(n - 1):
        _X = _AF(tf.matmul(_X, _W[i]) + _B[i])
    return tf.matmul(_X, _W[n - 1]) + _B[n - 1]

#Add L2 regularizers for the weight and bias matrices
#_W: The weight matrices
#_B: The bias matrices
#return: tensorflow variable representing l2 regularization cost
def _CreateL2Reg(_W, _B):
    n = len(_W)
    regularizers = tf.nn.l2_loss(_W[0]) + tf.nn.l2_loss(_B[0])
    for i in range(1, n):
        regularizers += tf.nn.l2_loss(_W[i]) + tf.nn.l2_loss(_B[i])
    return regularizers

#Create weight and bias vectors for an MLP
#layers: The number of neurons in each layer (including input and output)
#return: A tuple of lists of the weight and bias matrices respectively
def _CreateVars(layers):
    weight = []
    bias = []
    n = len(layers)
    for i in range(n - 1):
        #Fan-in for layer; used as standard dev
        lyrstd = np.sqrt(1.0 / layers[i])
        curW = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], stddev = lyrstd))
        weight.append(curW)
        curB = tf.Variable(tf.random_normal([layers[i + 1]], stddev = lyrstd))
        bias.append(curB)
    return (weight, bias)

#Helper function for selecting an activation function
#name: The name of the activation function
#return: A handle for the tensorflow activation function
def _GetActvFn(name):
    if name == 'tanh':
            return tf.tanh
    elif name == 'sig':
        return tf.sigmoid
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'relu6':
        return tf.nn.relu6
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'softsign':
        return tf.nn.softsign
    return None

#Helper function for getting a tensorflow optimizer
#name:    The name of the optimizer to use
#lr:      The learning rate if applicable
#return;  A the tensorflow optimization object
def _GetOptimizer(name, lr):
    if(name == 'adam'):
        return tf.train.AdamOptimizer(learning_rate = lr)
    elif(name == 'grad'):
        return tf.train.GradientDescentOptimizer(learning_rate = lr)
    elif(name == 'adagrad'):
        return tf.train.AdagradOptimizer(learning_rate = lr)
    elif(name == 'ftrl'):
        return tf.train.FtrlOptimizer(learning_rate = lr)
    return None

#Gives the next batch of samples of size self.batSz or the remaining
#samples if there are not that many
#A: Samples to choose from
#y: Targets to choose from
#cur: The next sample to use
#batSz: Size of the batch
#return: A tuple of the new samples and targets
def _NextBatch(A, y, cur, batSz):
    m = len(A)
    nxt = cur + batSz
    if(nxt > m):
        nxt = m
    return (A[cur:nxt], y[cur:nxt])

#Multi-Layer Perceptron for Regression
class MLPR:
    #Predicted outputs
    pred = None
    #The loss function
    loss = None
    #The optimization method
    optmzr = None
    #Max number of iterations
    mItr = None
    #Error tolerance
    tol = None
    #Tensorflow session
    sess = None
    #Tensorflow saver
    saver = None
    #Input placeholder
    x = None
    #Output placeholder
    y = None
    #Boolean for toggling verbose output
    vrbse = None
    #Batch size
    batSz = None
    prefix = ""

    #The constructor
    #param layers: A list of layer sizes
    #param actvFn: The activation function to use: 'tanh', 'sig', or 'relu'
    #param learnRate: The learning rate parameter
    #param decay: The decay parameter
    #param maxItr: Maximum number of training iterations
    #param tol: Maximum error tolerated
    #param batchSize: Size of training batches to use (use all if None)
    #param verbose: Print training information
    #param reg: Regularization weight
    def __init__(self, layers, actvFn = 'tanh', optmzr = 'adam', learnRate = 0.001, decay = 0.9,
                 maxItr = 2000, tol = 1e-2, batchSize = None, verbose = False, reg = 0.001, prefix=""):
        #Parameters
        self.tol = tol
        self.mItr = maxItr
        self.vrbse = verbose
        self.batSz = batchSize
        #Input size
        self.x = tf.placeholder("float", [None, layers[0]])
        #Output size
        self.y = tf.placeholder("float", [None, layers[-1]])
        #Setup the weight and bias variables
        weight, bias = _CreateVars(layers)
        #Create the tensorflow MLP model
        self.pred = _CreateMLP(self.x, weight, bias, _GetActvFn(actvFn))
        #Use L2 as the cost function
        self.loss = tf.reduce_mean(tf.nn.l2_loss(self.pred - self.y))
        #Use regularization to prevent over-fitting
        if(reg is not None):
            self.loss += _CreateL2Reg(weight, bias) * reg
        #Use ADAM method to minimize the loss function
        self.optmzr = _GetOptimizer(optmzr, learnRate).minimize(self.loss)
        #Initialize all variables on the TF session
        self.sess = tf.Session()
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)

    #Fit the MLP to the data
    #param A: numpy matrix where each row is a sample
    #param y: numpy matrix of target values
    def fit(self, A, y):
        m = len(A)
        #Begin training
        for i in range(self.mItr):
            #Batch mode or all at once
            if(self.batSz is None):
                self.sess.run(self.optmzr, feed_dict={self.x:A, self.y:y})
            else:
                for j in range(0, m, self.batSz):
                    batA, batY = _NextBatch(A, y, j, self.batSz)
                    self.sess.run(self.optmzr, feed_dict={self.x:batA, self.y:batY})
            err = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / m)
            if self.vrbse:
                print("Iter {:5d}\t{:.8f}".format(i + 1, err))
            if i % 1000 == 0 and i > 0:
                self.save("save_" + str(i) + ".ckpt")
            if(err < self.tol):
                break

    #Predict the output given the input (only run after calling fit)
    #param A: The input values for which to predict outputs
    #return: The predicted output values (one row per input sample)
    def predict(self, A):
        if(self.sess == None):
            print("Error: MLP has not yet been fitted.")
            return None
        res = self.sess.run(self.pred, feed_dict={self.x:A})
        return res

    #Predicts the ouputs for input A and then computes the RMSE between
    #The predicted values and the actualy values
    #param A: The input values for which to predict outputs
    #param y: The actual target values
    #return: The RMSE
    def score(self, A, y):
        scr = np.sqrt(self.sess.run(self.loss, feed_dict={self.x:A, self.y:y}) * 2.0 / len(A))
        return scr

    #Clean-up resources
    def __del__(self):
        self.sess.close()

    def save(self, fileName):
        savePath = self.saver.save(self.sess, "./" + prefix + "_" + fileName)
        print("Model saved in file: %s" % savePath)
        return savePath

    def load(self, fileName):
        #self.saver = tf.train.import_meta_graph(fileName + '.meta')
        self.saver.restore(self.sess, fileName)

class Tick:
    bid = None
    ask = None
    time = None

    def __init__(self, bid, ask, time):
        self.bid = bid
        self.ask = ask
        self.time = time

    def price(self):
        return (self.bid + self.ask) / 2.0

    def __unicode__(self):
        return 'Tick ' + str(self.time) + ': ' + str(self.bid) + ', ' + str(self.ask)

    def __str__(self):
        return self.__unicode__()

class TransationType:
    LONG, SHORT = range(2)

class Transaction:
    transaction_type = None
    open_price = None
    close_price = None
    stop_loss = None
    take_profit = None
    open_time = None
    close_time = None
    timeout = None

    def __init__(self, transaction_type, open_price, stop_loss, take_profit, open_time, timeout):
        self.transaction_type = transaction_type
        self.open_price = open_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.open_time = open_time
        self.timeout = timeout

    def close(self, price, time):
        self.close_price = price
        self.close_time = time

    def to_close(self, price, time):
        if time >= self.timeout:
            return True
        else:
            if self.transaction_type == TransationType.LONG:
                return price <= self.stop_loss or price >= self.take_profit
            else:
                return price >= self.stop_loss or price <= self.take_profit

    def profit(self):
        if self.close_price is None:
            return 0.0
        diff = self.close_price - self.open_price
        if self.transaction_type == TransationType.LONG:
            return diff
        else:
            return -diff

    def closed_on_tp(self):
        if self.close_price is None:
            return False
        if self.transaction_type == Transaction.LONG:
            return self.close_price >= self.take_profit
        else:
            return self.close_price <= self.take_profit

    def closed_on_sl(self):
        if self.close_price is None:
            return False
        if self.transaction_type == Transaction.LONG:
            return self.close_price <= self.stop_loss
        else:
            return self.close_price >= self.stop_loss

    def timedout(self, time):
        return self.timeout >= time

    def open(self):
        return self.close_time != None

    def __str__(self):
        return 'Trans ' + str(self.transaction_type) + \
            ', ot: ' + str(self.open_time) + \
            ', ct: ' + str(self.close_time) + \
            ', to: ' + str(self.timeout) + \
            ', o: ' + str(self.open_price) + \
            ', tp: ' + str(self.take_profit) + \
            ', c: ' + str(self.close_price) + \
            ', p: ' + str(self.profit())

class Trader:
    predictor = None
    filling_buffer = True

    open_transactions = []
    closed_transactions = []

    last_tick = None
    interval_start_time = None

    input_buffer = [] #10 x 10 min candlesticks
    interval_buffer = [] #ticks from current 10 min period

    INTERVAL_MINUTES = 10
    INPUT_BUFFER_SIZE = 10

    next_price_prediction = None

    STOP_LOSS_MARGIN = 0.0012
    PROFIT_TRESHOLD = 0.0030 #sprawdzic 8

    prediction_reached = False

    LAST_TICKS_MINUTES = 30
    last_ticks = []

    def __init__(self, predictor):
        self.predictor = predictor

    def support_levels(self):
        prices = [t.price() for t in self.last_ticks]
        #t0 = int(self.last_ticks[0].time.strftime('%s'))
        #time = [int(t.time.strftime('%s')) - t0 for t in self.last_ticks]
        #time = [t.time for t in self.last_ticks]
        maximas = [prices[i] for i in argrelextrema(np.asarray(prices), np.greater, order=7)[0]]
        minimas = [prices[i] for i in argrelextrema(np.asarray(prices), np.less, order=7)[0]]
        #for p in maximas:
        #    plt.axhline(y = p, color='r', linestyle='-')
        #for p in minimas:
        #    plt.axhline(y = p, color='g', linestyle='-')
        #plt.axhline(y = 1.102)
        #plt.plot([prices[i] for i in argrelextrema(np.asarray(prices), np.less)[0]])
        #plt.plot(time, prices)
        #plt.show()
        return minimas, maximas

    def __interval_ready(self):
        prices = [tick.price() for tick in self.interval_buffer]
        ahl = (np.max(prices) + np.min(prices))/2.0
        self.input_buffer.append(ahl)
        if len(self.input_buffer) == self.INPUT_BUFFER_SIZE:
            self.next_price_prediction = self.predictor.predict(np.asarray([self.input_buffer]))[0][0]
            self.input_buffer = self.input_buffer[1:]
            self.filling_buffer = False
        self.interval_start_time = None
        self.interval_buffer = []
        self.prediction_reached = False

    def tick_received(self, tick):
        price = tick.price()
        self.last_ticks.append(tick)
        i = 0
        while tick.time - self.last_ticks[i].time > dt.timedelta(minutes=self.LAST_TICKS_MINUTES):
            i += 1
        self.last_ticks = self.last_ticks[i:]
        if self.last_tick == None:
            self.last_tick = tick.time
        if self.interval_start_time == None:
            self.interval_start_time = tick.time
        if (tick.time - self.interval_start_time).seconds <= self.INTERVAL_MINUTES * 60:
            self.interval_buffer.append(tick)
        else:
            self.__interval_ready()
            self.interval_buffer = [tick]
            self.interval_start_time = tick.time
        self.last_tick = tick.time
        if not self.filling_buffer:
            #don't trade if predicted price was reached
            if abs(price - self.next_price_prediction) < 0.0001:
                self.prediction_reached = True
            """
            timedout = [t for t in self.open_transactions if t.timedout(tick.time)]
            for t in timedout:
                if (t.transaction_type == TransationType.LONG and t.take_profit <= self.next_price_prediction) or \
                    (t.transaction_type == TransationType.SHORT and t.take_profit >= self.next_price_prediction):
                    t.take_profit = self.next_price_prediction
                    t.timeout += dt.timedelta(minutes=10)
            """
            to_close = [t for t in self.open_transactions if t.to_close(price, tick.time)]
            for t in to_close:
                self.close_transaction(t, price)
            """
            for t in self.open_transactions:
                diff = t.take_profit - next_price_prediction
                if t.transaction_type == TransationType.LONG and diff < 0:
                    t.take_profit = next_price_prediction
                    if price - (t.stop_loss + diff) >= self.STOP_LOSS_MARGIN:
                        t.stop_loss += diff
                elif t.transaction_type == TransationType.SHORT and diff > 0:
                    t.take_profit = next_price_prediction
                    if (t.stop_loss - diff) - price >= self.STOP_LOSS_MARGIN:
                        t.stop_loss -= diff
            """
            treshold_reached = abs(price - self.next_price_prediction) >= self.PROFIT_TRESHOLD
            if treshold_reached and not self.prediction_reached and len(self.open_transactions) == 0:
                #if treshold_reached:
                open_transaction = False
                minimas, maximas = self.support_levels()
                if len(minimas) > 0 and price < self.next_price_prediction:
                    open_transaction = abs(price - np.min(minimas)) <= 0.00002
                elif len(maximas) > 0 and price > self.next_price_prediction:
                    open_transaction = abs(price - np.max(maximas)) <= 0.00002
                if open_transaction:
                    long_or_short = None
                    stop_loss = None
                    take_profit = self.next_price_prediction
                    if price < self.next_price_prediction:
                        long_or_short = TransationType.LONG
                        stop_loss = price - self.STOP_LOSS_MARGIN
                    elif price > self.next_price_prediction:
                        long_or_short = TransationType.SHORT
                        stop_loss = price + self.STOP_LOSS_MARGIN
                    if long_or_short != None:
                        t = Transaction(transaction_type = long_or_short,
                                        open_price = price,
                                        open_time = tick.time,
                                        stop_loss = stop_loss,
                                        take_profit = take_profit,
                                        timeout = tick.time + dt.timedelta(minutes=10))
                        #print "OPENING: ", tick.time, long_or_short, price, next_price_prediction, t
                        self.open_transaction(t)
                #print 'OPEN: ', t

    def open_transaction(self, t):
        #print 'OPEN:', t
        self.open_transactions.append(t)

    def close_transaction(self, t, price):
        self.open_transactions.remove(t)
        t.close(price = price, time = tick.time)
        self.closed_transactions.append(t)
        #print 'CLOSED:', t

class DataFeed:
    INPUT_DIR = './'
    files = []

    def __init__(self, dir='./'):
        self.INPUT_DIR = dir
        self.files = [self.INPUT_DIR + f
                    for f in sorted(os.listdir(self.INPUT_DIR))
                    if f.endswith(".csv") and os.path.isfile(os.path.join(self.INPUT_DIR, f))][-12:]

    def __datetime_converter(self,x):
        try:
            val = dt.datetime.strptime(x[:-3], '%Y-%m-%d %H:%M:%S.%f')
        except ValueError as e:
            val = dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
        return val - dt.timedelta(hours=17)

    def ticks(self):
        for f in self.files:
            with open(f,'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', )
                for row in reader:
                    time = self.__datetime_converter(row[3])
                    bid = float(row[4])
                    ask = float(row[5])
                    yield Tick(bid,ask,time)

feed = DataFeed(dir = './data/eur_usd/')

i = 10
#Number of neurons in the output layer
o = 1
#Number of neurons in the hidden layers
h = 64
#The list of layer sizes
layers = [i, h, h, h, h, o]
mlpr = MLPR(layers, optmzr='adagrad', maxItr = 30001, tol = 0.0001,
                reg = 0.01, decay=0.9, verbose = True, prefix="")
#Learn the data
#fileName = './save5.ckpt'
mlpr.load('./backup_z_amazonu/sth/test3_save_368000.ckpt')
np.set_printoptions(precision=8, suppress=True)
trader = Trader(mlpr)

A = np.loadtxt('./train_data/eur_usd/train_2016-01.csv',delimiter=',')
inputs = A[0][:10]

predicted = inputs

#for i in xrange(990):


A = A[:1000,[10]]
print A


"""
curr_time = None
days = 0
for tick in feed.ticks():
    if curr_time == None:
        curr_time = tick.time
    trader.tick_received(tick)
    if (tick.time - curr_time).days >= 1:
        days += 1
        print tick.time
        sum_ = sum([t.profit() for t in trader.closed_transactions]) * 10000.0
        print 'sum (pips):', sum_
        #print 'mean: ', np.mean([t.profit() for t in trader.closed_transactions])
        num_of_trans = len(trader.closed_transactions)
        if num_of_trans > 0:
            est_profit = sum_ * 10 - num_of_trans * 4
            print 'pip/day:', sum_/float(days)
            print '#:', num_of_trans
            print '#/day ', num_of_trans/float(days)
            print 'estimated profit ($):', est_profit
            print '$/day:', est_profit/float(days)
            profits = sorted([t.profit() for t in trader.closed_transactions])
            print 'min (pips):', profits[0] * 10000.0
            print 'max (pips):', profits[len(profits)-1] * 10000.0
            print 'mean (pips):', '{:.4f}'.format(np.mean(profits)*10000.0)
            print 'succ:', 100 * sum([1 for t in trader.closed_transactions if t.profit() > 0])/float(num_of_trans), '%'
            percentiles = [10,20,30,40,50,60,70,80,90,95,99]
            print np.column_stack((percentiles, np.percentile(profits, percentiles)))
        curr_time = tick.time
"""
