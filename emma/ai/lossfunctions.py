import keras.backend as K
import tensorflow as tf
from emma.utils.registry import lossfunctions, lossfunction
from emma.attacks.leakagemodels import LeakageModel
import sys


def get_loss(conf):
    if conf.loss_type in lossfunctions:
        f = lossfunctions[conf.loss_type]
        return f(conf)
    else:
        return conf.loss_type

# # My changes
# @lossfunction('correlation')
# def _get_correlation_loss(conf):
#     def correlation_loss(y_true_raw, y_pred_raw):
#         # print('##### debug #####')
#         # print(type(y_true_raw))
#         # print(type(y_pred_raw))
#         # print(tf.shape(y_true_raw, out_type=tf.dtypes.int32, name=None))
#         # print(tf.shape(y_pred_raw, out_type=tf.dtypes.int32, name=None))
#         # breakpoint()
#
#         """
#         Custom loss function that calculates the Pearson correlation of the prediction with
#         the true values over a number of batches.
#         """
#
#         # print("Debug message for printing the size of the encodings and the intermediate values")
#         # print(y_true_raw.shape)
#         # print(type(y_true_raw))
#         # print(y_pred_raw.shape)
#         # print(type(y_pred_raw))
#         # breakpoint()
#
#         y_true = (y_true_raw - K.mean(y_true_raw, axis=0,
#                                       keepdims=True))  # We are taking correlation over columns, so normalize columns
#         y_pred = (y_pred_raw - K.mean(y_pred_raw, axis=0, keepdims=True))
#
#         # loss = K.variable(0.0)
#
#         loss = 0
#         for key_col in range(0, conf.key_high - conf.key_low):  # 0 - 16
#             y_key = K.expand_dims(y_true[:, key_col], axis=1)  # [?, 16] -> [?, 1]
#             y_keypred = K.expand_dims(y_pred[:, key_col], axis=1)  # [?, 16] -> [?, 1]
#             denom = K.sqrt(K.dot(K.transpose(y_keypred), y_keypred)) * K.sqrt(K.dot(K.transpose(y_key), y_key))
#             denom = K.maximum(denom, K.epsilon())
#             correlation = K.dot(K.transpose(y_key), y_keypred) / denom
#             # correlation = tf.squeeze(correlation)
#             # correlation = K.reshape(correlation, (-1, 0))
#             loss = 1.0 - correlation
#             # loss.assign_add(1.0 - correlation)
#
#         # loss = 0
#         # y_key = K.expand_dims(y_true[0, 0], axis=1)  # [?, 16] -> [?, 1]
#         # y_keypred = K.expand_dims(y_pred[0, 0], axis=1)  # [?, 16] -> [?, 1]
#         # denom = K.sqrt(K.dot(K.transpose(y_keypred), y_keypred)) * K.sqrt(K.dot(K.transpose(y_key), y_key))
#         # denom = K.maximum(denom, K.epsilon())
#         # correlation = K.dot(K.transpose(y_key), y_keypred) / denom
#         # # correlation = tf.squeeze(correlation)
#         # # correlation = K.reshape(correlation, (-1, 0))
#         # loss = 1.0 - correlation
#         # # loss.assign_add(1.0 - correlation)
#         print('loss', loss)
#         print(y_true[:, 0])
#         # breakpoint()
#
#         return loss
#
#     return correlation_loss

# Correct loss funtion
@lossfunction('correlation')
def _get_correlation_loss(conf):
    def correlation_loss(y_true_raw, y_pred_raw):
        # print('##### debug #####')
        # print(type(y_true_raw))
        # print(type(y_pred_raw))
        # print(tf.shape(y_true_raw, out_type=tf.dtypes.int32, name=None))
        # print(tf.shape(y_pred_raw, out_type=tf.dtypes.int32, name=None))
        # breakpoint()

        """
        Custom loss function that calculates the Pearson correlation of the prediction with
        the true values over a number of batches.
        """

        # print("Debug message for printing the size of the encodings and the intermediate values")
        # print(y_true_raw.shape)
        # print(type(y_true_raw))
        # print(y_pred_raw.shape)
        # print(type(y_pred_raw))
        # breakpoint()

        y_true = (y_true_raw - K.mean(y_true_raw, axis=0,
                                      keepdims=True))  # We are taking correlation over columns, so normalize columns
        y_pred = (y_pred_raw - K.mean(y_pred_raw, axis=0, keepdims=True))

        # loss = K.variable(0.0)

        loss = 0
        for key_col in range(0, conf.key_high - conf.key_low):  # 0 - 16
            y_key = K.expand_dims(y_true[:, key_col], axis=1)  # [?, 16] -> [?, 1]
            y_keypred = K.expand_dims(y_pred[:, key_col], axis=1)  # [?, 16] -> [?, 1]
            denom = K.sqrt(K.dot(K.transpose(y_keypred), y_keypred)) * K.sqrt(K.dot(K.transpose(y_key), y_key))
            denom = K.maximum(denom, K.epsilon())
            correlation = K.dot(K.transpose(y_key), y_keypred) / denom
            # correlation = tf.squeeze(correlation)
            # correlation = K.reshape(correlation, (-1, 0))
            loss = 1.0 - correlation
            # loss.assign_add(1.0 - correlation)
            # print('DEbug')
            # print(loss.numpy())
            # breakpoint()
        return loss

    return correlation_loss


@lossfunction('correlation_special')
def _get_special_correlation_loss(conf):
    def correlation_loss(y_true_raw, y_pred_raw):
        """
        Custom loss function that calculates the Pearson correlation of the prediction with
        the true values over a number of batches, with the addition of a weight parameter that
        is used to approximate the true key byte value.
        """
        y_true = (y_true_raw - K.mean(y_true_raw, axis=0,
                                      keepdims=True))  # We are taking correlation over columns, so normalize columns
        y_pred = (y_pred_raw - K.mean(y_pred_raw, axis=0, keepdims=True))
        weight_index = conf.key_high - conf.key_low

        loss = K.variable(0.0)
        weight = K.mean(y_pred_raw[:, weight_index], axis=0)
        for key_col in range(0, conf.key_high - conf.key_low):  # 0 - 16
            y_key = K.expand_dims(y_true[:, key_col], axis=1)  # [?, 16] -> [?, 1]
            y_keypred = K.expand_dims(y_pred[:, key_col], axis=1)  # [?, 16] -> [?, 1]
            denom = K.sqrt(K.dot(K.transpose(y_keypred), y_keypred)) * K.sqrt(K.dot(K.transpose(y_key), y_key))
            denom = K.maximum(denom, K.epsilon())
            correlation = K.dot(K.transpose(y_key), y_keypred) / denom
            loss += 1.0 - correlation

            alpha = 0.001
            exact_pwr = tf.multiply(weight, y_pred_raw[:, key_col])
            loss += alpha * K.sum(K.square(y_true_raw[:, key_col] - exact_pwr))

        return loss

    return correlation_loss


@lossfunction('abs_distance')
def _get_abs_distance_loss(conf):
    return __get_distance_loss(conf, False)


@lossfunction('squared_distance')
def _get_squared_distance_loss(conf):
    return __get_distance_loss(conf, True)


def __get_distance_loss(conf, squared):
    def distance_loss(y_true_raw, y_pred_raw):
        y_true = y_true_raw
        y_pred = y_pred_raw

        loss = K.variable(0.0)
        for key_col in range(0, conf.key_high - conf.key_low):  # 0 - 16
            y_key = y_true[:, key_col]  # [?, 16] -> [?,]
            y_keypred = y_pred[:, key_col]  # [?, 16] -> [?,]
            if squared:
                loss += K.sum(K.square(y_key - y_keypred))
            else:
                loss += K.sum(K.abs(y_key - y_keypred))

        return loss

    return distance_loss


@lossfunction('softmax_crossentropy')
def _get_crossentropy_loss(*_):
    def crossentropy_loss(y_true, y_pred):
        y_true = tf.stop_gradient(y_true)
        return tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

    return crossentropy_loss
