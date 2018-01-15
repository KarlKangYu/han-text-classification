import tensorflow as tf

class HanModel(object):
    def __init__(self, sent_len, doc_len, vocab_size, embed_size,
                 learning_rate, keep_prob, word_hidden_size,
                 word_attention_size, sent_hidden_size, sent_attention_size):
        self.input_X = tf.placeholder(tf.int32,
                                      shape=(None, doc_len, sent_len))
        self.input_y = tf.placeholder(tf.float32, shape=(None,))
        self.vocab_size = vocab_size
        self.embedding_size = embed_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.word_hidden_size = word_hidden_size
        self.word_attention_size = word_attention_size
        self.sent_hidden_size = sent_hidden_size
        self.sent_attention_size = sent_attention_size

    def training(self, loss, global_step):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss, global_step)
        return train_op

    def inference(self, inputs):
        # inputs.shape=(batch_size, doc_len, sent_len, embedding_size)
        # Batch size of sentence level is batch_size*doc_len.
        sent_batch_size = inputs.shape[0].value * inputs.shape[1].value
        sents = tf.reshape(
            inputs,
            [sent_batch_size, inputs.shape[2].value, self.embedding_size])
        # word_contexts.shape=(sent_batch_size, 2*word_hidden_size)
        word_contexts = self._encoder_attention(
            sents, self.word_hidden_size, self.word_attention_size)
        

    def loss(self):
        pass

    def evaluate(self):
        pass

    def embedding_from_scratch(self):
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            embeddings = tf.Variable(
                tf.random_uniform(
                    [self.vocab_size, self.embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self.input_X)
            return embed

    def _attention(self, inputs, attention_size):
        # inputs.shape=(batch_size, max_time, input_size)
        # Return the int value of the Dimension object.
        input_size = inputs.shape[2].value
        W = tf.Variable(
            tf.random_normal(
                [input_size, attention_size], dtype=tf.float32))
        b = tf.Variable(
            tf.random_normal([attention_size], dtype=tf.float32))
        # u.shape=(batch_size, max_time, attention_size)
        u = tf.tanh(tf.matmul(inputs, W) + b)
        context = tf.Variable(
            tf.random_normal([attention_size], dtype=tf.float32))
        # The inputs of tf.matmul must be tensors of rank >= 2
        # where the inner 2 dimensions specify valid matrix
        # multiplication arguments, and any further outer dimensions match.
        # Hence, tf.matmul(u, context) will throw an ValueError exception.
        # Here, I use tf.tensordot instead.
        # logits.shape=(batch_size, max_time)
        logits = tf.tensordot(u, context, axes=1)
        # alpha.shape=(batch_size, max_time)
        alpha = tf.nn.softmax(logits=logits)
        # Sum of array elements over the time axis.
        # sent_vec.shape=(batch_size, input_size)
        sent_vec = tf.resuce_sum(inputs * tf.expand_dims(alpha, -1), 1)
        return sent_vec, alpha

    def _encoder(self, inputs, hidden_size):
        # inputs.shape=(batch_size, max_time, input_size)
        fw_gru_cell = tf.contrib.rnn.GruCell(num_units=hidden_size)
        bw_gru_cell = tf.contrib.rnn.GruCell(num_units=hidden_size)
        # outputs is a tuple (output_fw, output_bw).
        # output_fw.shape=(batch_size, max_time, hidden_size)
        # output_bw.shape=(batch_size, max_time, hidden_size)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            fw_gru_cell, bw_gru_cell, inputs)
        # annotations.shape=(batch_size, max_time, 2*hidden_size)
        annotations = tf.concat(outputs, 2)
        return annotations

    def _encoder_attention(self, inputs, hidden_size, attention_size):
        annotations = self._encoder(inputs, hidden_size)
        return self._attention(annotations, attention_size)
        