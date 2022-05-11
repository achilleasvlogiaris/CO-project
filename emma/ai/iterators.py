"""
AI Iterators: classes that prepare the inputs and labels for the models in models.py.
"""

from emma.io.emresult import EMResult
from emma.emcap.streamserver import StreamServer
from celery.utils.log import get_task_logger
# from keras.utils import to_categorical
from emma.io.traceset import TraceSet
from emma.io.dataset import get_dataset_normalization_mean_std
from emma.attacks.leakagemodels import LeakageModel
from emma.ai.inputs import AIInput
from emma.utils.utils import shuffle_random_multiple, EMMAException

import numpy as np
from emma.processing import ops

logger = get_task_logger(__name__)


class AISignalIteratorBase():
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None, stream_server=None):
        self.trace_set_paths = trace_set_paths
        self.conf = conf
        self.batch_size = batch_size
        self.cache = {}
        self.index = 0
        self.values_batch = []
        self.signals_batch = []
        self.request_id = request_id
        self.max_cache = conf.max_cache
        self.augment_roll = conf.augment_roll
        self.augment_noise = conf.augment_noise
        self.augment_shuffle = conf.augment_shuffle
        self.normalize = conf.normalize
        self.stream_server = stream_server
        self.traces_per_set = conf.traces_per_set
        self.num_total_examples = 0
        if conf.online:
            self.num_total_examples = batch_size
        else:
            self.num_total_examples = len(self.trace_set_paths) * self.traces_per_set
        logger.info("Iterator contains %d examples" % self.num_total_examples)

    def __iter__(self):
        return self

    def get_all_as_trace_set(self, limit=None):
        if limit is None:
            traces_to_get = self.trace_set_paths
            print('traces_to_get', traces_to_get)
        else:
            traces_to_get = self.trace_set_paths[0:limit]

        result = EMResult(task_id=self.request_id)  # Make new collection of results
        ops.process_trace_set_paths(result, traces_to_get, self.conf, keep_trace_sets=True, request_id=self.request_id)  # Store processed trace path in result

        all_traces = []
        for trace_set in result.trace_sets:
            all_traces.extend(trace_set.traces)

        result = TraceSet(name="all_traces")
        result.set_traces(all_traces)

        return result

    def _preprocess_trace_set(self, trace_set):
        # X
        signals = np.array([trace.signal for trace in trace_set.traces], dtype=float)

        # Y
        values = np.array([trace.plaintext for trace in trace_set.traces], dtype=float)

        return signals, values

    def fetch_features(self, trace_set_path):
        """
        Fethes the features (raw trace and y-values) for a single trace path.
        """
        # Memoize
        if trace_set_path in self.cache:
            return self.cache[trace_set_path]

        # Apply actions from work()
        result = EMResult(task_id=self.request_id)  # Make new collection of results
        ops.process_trace_set_paths(result, [trace_set_path], self.conf, keep_trace_sets=True, request_id=self.request_id)  # Store processed trace path in result

        if len(result.trace_sets) > 0:
            signals, values = self._preprocess_trace_set(result.trace_sets[0])  # Since we iterate per path, there will be only 1 result in trace_sets

            # Cache
            if (self.max_cache is None) or len(self.cache.keys()) < self.max_cache:
                self.cache[trace_set_path] = (signals, values)

            return signals, values
        else:
            return None

    def fetch_features_online(self):
        logger.debug("Stream: waiting for packet in queue")
        # Get from blocking queue
        trace_set = self.stream_server.queue.get()

        # Apply work()
        logger.debug("Stream: processing trace set")
        result = EMResult(task_id=self.request_id)
        ops.process_trace_set(result, trace_set, self.conf, keep_trace_sets=False, request_id=self.request_id)

        # Get signals and values
        signals, values = self._preprocess_trace_set(trace_set)

        return signals, values

    def _augment_roll(self, signals, roll_limit=None):  # TODO unit test!
        roll_limit = roll_limit if not roll_limit is None else len(signals[0,:])
        roll_limit_start = -roll_limit if not roll_limit is None else 0
        logger.debug("Data augmentation: rolling signals")
        num_signals, signal_len = signals.shape
        for i in range(0, num_signals):
            signals[i,:] = np.roll(signals[i,:], np.random.randint(roll_limit_start, roll_limit))
        return signals

    def _augment_noise(self, signals, mean=0.0, std=1.0):
        logger.debug("Data augmentation: adding noise to signals")
        num_signals, signal_len = signals.shape
        for i in range(0, num_signals):
            signals[i,:] = signals[i,:] + np.random.normal(loc=mean, scale=std, size=signal_len)
        return signals

    def _normalize(self, signals):
        logger.debug("Normalizing data")
        mean, std = get_dataset_normalization_mean_std(self.conf.dataset_id)
        num_signals, signal_len = signals.shape
        for i in range(0, num_signals):
            signals[i,:] = (signals[i,:] - mean) / std
        return signals

    def next(self):
        # Bound checking
        if self.index < 0 or self.index >= len(self.trace_set_paths):
            raise EMMAException("ERROR: index is %d but length is %d" % (self.index, len(self.trace_set_paths)))

        while True:
            # Do we have enough samples in buffer already?
            if len(self.signals_batch) > self.batch_size:
                # Get exactly batch_size training examples
                signals_return_batch = np.array(self.signals_batch[0:self.batch_size])
                values_return_batch = np.array(self.values_batch[0:self.batch_size])

                # Keep the remainder for next iteration
                self.signals_batch = self.signals_batch[self.batch_size:]
                self.values_batch = self.values_batch[self.batch_size:]

                # Return
                return signals_return_batch,values_return_batch

            # Determine next trace set path
            trace_set_path = self.trace_set_paths[self.index]
            self.index += 1
            if self.index >= len(self.trace_set_paths):
                self.index = 0

            # Fetch features from online stream or from a path
            if self.conf.online:
                result = self.fetch_features_online()
            else:
                result = self.fetch_features(trace_set_path)
            if result is None:
                continue
            signals, values = result

            # Normalize
            if self.normalize:
                signals = self._normalize(signals)

            # Augment if enabled
            if self.augment_roll:
                signals = self._augment_roll(signals, roll_limit=16)
            if self.augment_noise:
                signals = self._augment_noise(signals, mean=0.0, std=0.01)
            if self.augment_shuffle:
                signals, values = shuffle_random_multiple([signals, values])

            # Concatenate arrays until batch obtained
            self.signals_batch.extend(signals)
            self.values_batch.extend(values)

    def __next__(self):
        return self.next()


class AICorrSignalIterator(AISignalIteratorBase):
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None, stream_server=None):
        super(AICorrSignalIterator, self).__init__(trace_set_paths, conf, batch_size, request_id, stream_server)

    def _preprocess_trace_set(self, trace_set):
        """
        Preprocess trace_set specifically for AICorrNet
        """

        # Get model inputs (usually the trace signal)
        signals = AIInput(self.conf).get_trace_set_inputs(trace_set)

        # Get model labels (key byte leakage values to correlate / analyze)
        values = LeakageModel(self.conf).get_trace_set_leakages(trace_set)

        return signals, values


class AutoEncoderSignalIterator(AISignalIteratorBase):
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None, stream_server=None):
        super(AutoEncoderSignalIterator, self).__init__(trace_set_paths, conf, batch_size, request_id, stream_server)

    def _preprocess_trace_set(self, trace_set):
        """
        Preprocess trace_set specifically for AutoEncoder
        """

        # Get model inputs (usually the trace signal)
        signals = AIInput(self.conf).get_trace_set_inputs(trace_set)

        return signals, signals  # Model output is same as model input


class AISHACPUSignalIterator(AISignalIteratorBase):
    def __init__(self, trace_set_paths, conf, batch_size=10000, request_id=None, stream_server=None, hamming=True, subtype='vgg16'):
        super(AISHACPUSignalIterator, self).__init__(trace_set_paths, conf, batch_size, request_id, stream_server=None)
        self.hamming = hamming
        self.subtype = subtype

    def _adapt_input_vgg(self, traces):
        batch = []
        for trace in traces:
            side_len = int(np.sqrt(len(trace.signal) / 3.0))
            max_len = side_len * side_len * 3
            image = np.array(trace.signal[0:max_len], dtype=float).reshape(side_len, side_len, 3)
            batch.append(image)
        return np.array(batch)

    def _preprocess_trace_set(self, trace_set):
        """
        Preprocessing specifically for AISHACPU
        """

        # Get training data
        if self.subtype == 'vgg16':
            signals = self._adapt_input_vgg(trace_set.traces)
        else:
            signals = np.array([trace.signal for trace in trace_set.traces], dtype=float)

        # Get one-hot labels (bytes XORed with 0x36)
        if self.hamming:
            values = np.zeros((len(trace_set.traces), 9), dtype=float)
        else:
            values = np.zeros((len(trace_set.traces), 256), dtype=float)
        index_to_find = 0  # Byte index of SHA-1 key
        for i in range(len(trace_set.traces)):
            trace = trace_set.traces[i]
            key_byte = trace.plaintext[index_to_find]
            if self.hamming:
                values[i, hw[key_byte ^ 0x36]] = 1.0
            else:
                values[i, key_byte ^ 0x36] = 1.0

        return signals, values


def get_iterators_for_model(model_type, training_trace_set_paths, validation_trace_set_paths, conf, batch_size=512, hamming=False, subtype='custom', request_id=None):
    # Stream samples from other machine?
    if conf.online:
        stream_server = StreamServer(conf)
        batch_size = 32
    else:
        stream_server = None
        batch_size = conf.batch_size

    training_iterator = None
    validation_iterator = None
    if model_type == 'aicorrnet' or model_type == 'aiascad':  # TODO unify with one iterator
        training_iterator = AICorrSignalIterator(training_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server)
        validation_iterator = AICorrSignalIterator(validation_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server)
    elif model_type == 'aishacpu':
        training_iterator = AISHACPUSignalIterator(training_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server, hamming=hamming, subtype=subtype)
        validation_iterator = AISHACPUSignalIterator(validation_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server, hamming=hamming, subtype=subtype)
    elif model_type == 'aishacc':
        training_iterator = AISHACPUSignalIterator(training_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server, hamming=hamming, subtype='custom')
        validation_iterator = AISHACPUSignalIterator(validation_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server, hamming=hamming, subtype='custom')
    elif model_type == 'autoenc':
        training_iterator = AutoEncoderSignalIterator(training_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server)
        validation_iterator = AutoEncoderSignalIterator(validation_trace_set_paths, conf, batch_size=batch_size, request_id=request_id, stream_server=stream_server)
    else:
        logger.error("Unknown training procedure %s specified." % model_type)
        exit(1)

    return training_iterator, validation_iterator
