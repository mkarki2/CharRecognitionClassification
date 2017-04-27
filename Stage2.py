'''
Imports
'''
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.gof import graph
import matplotlib.pyplot as plt
from rbm import RBM
from parameters_file import tic, toc
# import theano.sandbox.cuda
# gpu = "gpu2"
# theano.sandbox.cuda.use(gpu)

class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        """
        :input:  input of the architecture (one mini-batch).
        :n_in: The number of input units, the dimension of the data space.
        :n_out: The number of output units, the dimension of the labels (here it's one).
        """

        # Initialize the weights to be all zeros.
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                               name='b',
                               borrow=True)

        # p_y_given_x forms a matrix, and y_pred will extract first element from each list.
        self.p_y_given_x = T.dot(input, self.W) + self.b

        # This caused a lot of confusion! It's basically the difference between [x] and x in python.
        self.y_pred = self.p_y_given_x[:, :]

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

    def squared_errors(self,y):
        """ Returns the (max of ) mean of squared errors of the linear regression on this data. """
        return  (T.mean(T.sqr(self.y_pred - y),axis=0)).max()

#From the Deeplearning.net library

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            # for i in range(len(y)):
            #     if y ==0

            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

# If dropout is used this function is called
def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

class HiddenLayer(object):
    """
    Hidden Layer class for a Multi-Layer Perceptron. This is exactly the same as the reference
    code from the documentation, except for T.sigmoid instead of T.tanh.
    """

    def __init__(self, rng, input, n_in, n_out,dropout_rate, W=None, b=None, activation=T.tanh):
        """
        :rng: A random number generator for initializing weights.
        :input: A symbolic tensor of shape (n_examples, n_in).
        :n_in: Dimensionality of input.
        :n_out: Number of hidden units.
        :activation: Non-linearity to be applied in the hidden layer.
        """

        # W is initialized with W_values, according to the "Xavier method".
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        # Initialize the bias weights.
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # The output of all the inputs, "squashed" via the activation function.
        lin_output = T.dot(input, self.W) + self.b
        self.output = lin_output if activation is None else activation(lin_output)
        if dropout_rate>0.0:
            self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)

        # Miscellaneous stuff
        self.params = [self.W, self.b]
        self.input = input

#Convert dataset to shared dataset so that it can be transferred to GPU
def convert_dataset(dataset,c):
    train_set, valid_set= dataset[0], dataset[1]
    assert (train_set[0].shape)[1] == (valid_set[0].shape)[1], \
        "Number of features for train,val do not match: {} and {}.".format(train_set.shape[1], valid_set.shape[1])

    def shared_dataset(data_xy, borrow=True,classification=c):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        if classification:
            shared_y = T.cast(shared_y, 'int32')
        return shared_x,shared_y

    tic()

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    # test_set_x, test_set_y = shared_dataset(test_set)
    toc('Dataset Converted to shared dataset.')

    num_features = (train_set[0].shape)[1]
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]#, (test_set_x, test_set_y)]
    return rval, num_features

#DBN class (the model we train)
class DBN(object):
    def __init__(self, numpy_rng, n_ins, params,
                 theano_rng=None):

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(params.layer_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as a matrix
        if params.classification:
            self.y = T.ivector('y')  # the labels are presented as 1D vector
        # of labels
        else:
            self.y = T.matrix('y')

        for i in range(self.n_layers):
            # construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = params.layer_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=params.layer_sizes[i],
                                        dropout_rate=params.dropout,
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=params.layer_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # self.decisionLayer = HiddenLayer(rng=numpy_rng,input=self.sigmoid_layers[-1].output,
        #                              n_in=hidden_layers_sizes[-1],
        #                              n_out=n_outs,
        #                              activation=T.nnet.relu)

        if (params.classification):

            self.decisionLayer = LogisticRegression(input=self.sigmoid_layers[-1].output,
                                         n_in=params.layer_sizes[-1],
                                         n_out=params.num_output,
                                         )
            self.errors = self.decisionLayer.negative_log_likelihood

        else:
            self.decisionLayer = LinearRegression(input=self.sigmoid_layers[-1].output,
                                     n_in=params.layer_sizes[-1],
                                     n_out=params.num_output,
                                     )
            self.errors = self.decisionLayer.squared_errors

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.decisionLayer.W).sum()
        for i in range(2 * self.n_layers)[0::2]:
            self.L1 += abs(self.params[i]).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.decisionLayer.W ** 2).sum()
        for i in range(2 * self.n_layers)[0::2]:
            self.L2_sqr += (self.params[i] ** 2).sum()

        self.finetune_cost = (self.errors(self.y))+ params.L1_reg * self.L1 + params.L2_reg * self.L2_sqr

        self.params.extend(self.decisionLayer.params)
        self.input = input

        self.y_pred = self.decisionLayer.y_pred

    def pretrain_setup(self, train_set_x, batch_size, k):

        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * int(batch_size/4)
        # ending of a batch given `index`
        batch_end = batch_begin + int(batch_size/4)


        pretrain_fns = []
        for rbm in self.rbm_layers:
            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            persistent_chain = theano.shared(numpy.zeros((batch_size,
                                                          rbm.n_hidden),
                                                         dtype=theano.config.floatX))
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k) #persisetnet=None
            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def finetune_setup(self, datasets, batch_size):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        # (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= (batch_size)
        # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        # n_test_batches /= batch_size

        index = T.lscalar()  # index to a [mini]batch
        lr = T.scalar('lr', dtype=theano.config.floatX)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * lr))

        train_fn = theano.function(
            inputs=[index,lr],
            outputs=[self.finetune_cost, self.errors(self.y)],
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }

        )

        # test_score_i = theano.function(
        #     [index],
        #     self.errors(self.y),
        #     givens={
        #         self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
        #         self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
        #     }
        # )

        valid_score_i = theano.function(
            [index],
            self.errors(self.y),
            givens={
                self.x: valid_set_x[index * int(batch_size): (index + 1) * int(batch_size)],
                self.y: valid_set_y[index * int(batch_size): (index + 1) * int(batch_size)]
            }
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(int(n_valid_batches))]

        # Create a function that scans the entire test set
        # def test_score():
        #     return [test_score_i(i) for i in range(int(n_test_batches))]

        return train_fn, valid_score #, test_score

#main function of this file
def test_DBN(best_mse,network,dataset, params):



    datasets, features = convert_dataset(dataset,params.classification)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / params.batch_size

    if params.finetune_var.retrain==False:
        numpy_rng = numpy.random.RandomState(125)
        print('... building the model')

        # construct the Deep Belief Network
        dbn = DBN(numpy_rng=numpy_rng, n_ins=features,params=params)

        # setup the pretraining
        print('... getting the pretraining functions')
        pretraining_fns = dbn.pretrain_setup(train_set_x=train_set_x, batch_size=params.batch_size, k=params.pretrain_var.k)
        dbn= pretrain(pretraining_fns, params.pretrain_var, n_train_batches, dbn)
    else:
         if network:
            dbn=network
         else:
            model_file = open(params.finetune_var.model_name, 'rb')
            dbn = pickle.load(model_file)
            model_file.close()


    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, valid_score = dbn.finetune_setup(
            batch_size=params.batch_size,
            datasets=datasets)
    dbn,final_mse=train(best_mse,n_train_batches,params.finetune_var,train_fn, valid_score,dbn)

    # save the last model
    # with open('final' + finetune_var.model_name, 'wb') as f2:
    #     pickle.dump(dbn, f2, protocol=pickle.HIGHEST_PROTOCOL)
    #     print('Last Model Saved as: ' + 'final' + finetune_var.model_name)
    train_set_x.set_value([[]])
    datasets[0][0].set_value([[]])
    datasets[1][0].set_value([[]])
    datasets[0]=[]
    datasets[1]=[]
    # datasets[2]=[]
    # import gc
    # gc.collect()
    return  dbn,final_mse

def pretrain(pretraining_fns, pretrain_var, n_train_batches, dbn):

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pretraining epochs
        for epoch in range(pretrain_var.epochs):
            # go through the training set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_var.lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), )
            print(numpy.mean(c))

    end_time = timeit.default_timer()

    print(('The pretraining code in file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    return dbn

def train(best_mse,n_train_batches,finetune_var,train_fn, valid_score,dbn):
    # patience = 25
    # patience_increase = 10
    # improvement_threshold = 0.99995
    # validation_frequency = int(min(n_train_batches, patience / 2))
    validation_frequency = int(n_train_batches)
    best_validation_loss = best_mse
    test_error = 0.
    # start_time = timeit.default_timer()
    patience2 = 500

    done_looping = False
    epoch = 0
    k123=50

    # lr=finetune_var.lr
    while (epoch < finetune_var.epochs) and (not done_looping):
        epoch = epoch + 1
        if epoch % k123 == 0:
            finetune_var.lr *= 0.995
            print('New Finetune LR is: %f.'%finetune_var.lr)
        finetune_var.lr=max(0.01,finetune_var.lr)
        if patience2 <= 0:
            done_looping = True
            break

        for minibatch_index in range(int(n_train_batches)):

            minibatch_finetune_cost, minibatch_mse_cost = train_fn(minibatch_index,finetune_var.lr,)
            iter = (epoch - 1) * int(n_train_batches) + minibatch_index

            if (iter + 1) % validation_frequency == 0:

                validation_losses = valid_score()
                this_validation_loss = numpy.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error: %f , finetune_cost: %f, train error: %f'
                    % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss*100,
                        minibatch_finetune_cost*100, numpy.mean(minibatch_mse_cost)*100
                    )
                )

                # done_looping = False

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    patience2=10000
                    # # improve patience if loss improvement is good enough
                    # if (
                    #             this_validation_loss < best_validation_loss *
                    #             improvement_threshold
                    # ):
                    #      patience = max(patience, iter%(n_train_batches) + patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss

                    # test it on the test set
                    # test_losses = test_score()
                    # test_error = numpy.mean(test_losses)
                    # print(('     \tepoch %i,  minibatch %i/%i, test Accuracy: '
                    #        'best model %f') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_error*100))


                    # save the best model
                    with open(finetune_var.model_name, 'wb') as f:
                        pickle.dump(dbn, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print('         \tModel Saved!')

                else:
                    patience2=patience2-1


                        # if patience <= iter:
                        #     done_looping = True
                        #
                        #     break


    # end_time = timeit.default_timer()
    # print(
    #     (
    #         'Optimization complete with best validation score of %f, '
    #         'with test performance %f'
    #     ) % (best_validation_loss, test_error)
    # )
    # print(('The fine tuning code in file ' +
    #        os.path.split(__file__)[1] +
    #        ' ran for %.2fm' % ((end_time - start_time)
    #                            / 60.)), file=sys.stderr)
    # print('Best Model Saved as: ' + finetune_var.model_name)
    with open(finetune_var.model_name+'last', 'wb') as f:

                        pickle.dump(dbn, f, protocol=pickle.HIGHEST_PROTOCOL)

    return dbn,best_validation_loss

def predict(X_test, filename='best_model_actual_data.pkl'):
    # load the saved model
    model_file = open(filename, 'rb')
    classifier = pickle.load(model_file)
    model_file.close()
    y_pred = classifier.y_pred

    # find the input to theano graph
    inputs = graph.inputs([y_pred])
    # select only x
    inputs = [item for item in inputs if item.name == 'x']
    # compile a predictor function
    predict_model = theano.function(
        inputs=inputs,
        outputs=y_pred)

    predicted_values = predict_model(X_test.astype(numpy.float32))

    return predicted_values

