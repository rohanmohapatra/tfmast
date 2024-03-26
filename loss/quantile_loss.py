import tensorflow as tf


class QuantileLoss:
    def __init__(self, quantiles: list, n_targets: int):
        self._quantiles = quantiles
        self._n_targets = n_targets

    def quantile_loss(self, y_true, y_pred):
        """
        Calculate quantile loss. Here, y_true.shape = batch_sz, n_dec_steps, n_targets
        whereas y_pred.shape = batch_sz, n_dec_steps, n_targets * n_quantiles.

        So, in the output you expect sequences of n_targets columns for each quantile.
        """
        y_true = tf.cast(y_true, tf.float32)
        loss = 0.0
        for i, q in enumerate(self._quantiles):
            loss += self._q_loss(
                y_true, y_pred[..., self._n_targets * i : self._n_targets * (i + 1)], q
            )

        return loss

    def _q_loss(self, y_true, y_pred, quantile: float):
        """
        For a given quantile, calculate error for each target parameter.
        Optionally, you can mask and omit certain values from the
        loss calculation, as you would do with pad token in NLP.
        """
        assert 0 < quantile < 1
        err = y_true - y_pred

        # a * (y - y_hat)           for y_hat <= y
        # (1 - a) * (y_hat - y)     for y_hat > y
        q_loss = quantile * tf.maximum(err, 0.0) + (1.0 - quantile) * tf.maximum(
            -err, 0.0
        )

        return tf.reduce_sum(input_tensor=q_loss, axis=-1)
