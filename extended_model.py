from simple_model import S2SAttentionModel

import tensorflow as tf


class RandRegModel(S2SAttentionModel):
    def get_embeddings(self, inputs, embedding_size):
        embeddings = tf.Variable(
            tf.random_uniform([self.V, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        return embed

    def get_representations(self):
        # embedding
        if self.args.use_embedding:
            primitive = self.get_embeddings(self.x, self.args.embedding_size)
            function = self.get_embeddings(self.x, self.args.function_embedding_size)

        # compute switch
        with tf.variable_scope("compute_switch"):
            switch_score = self.get_embeddings(self.x, 1)
            self.switch = tf.nn.sigmoid(
                switch_score / self.args.switch_temperature)
        if not self.args.remove_switch:
            if self.args.relu_switch:
                switch_primitive = tf.nn.relu(2 * self.switch - 1)
                switch_function = tf.nn.relu(1 - 2 * self.switch)
            else:
                switch_primitive = self.switch
                switch_function = 1 - self.switch
            primitive = tf.multiply(switch_primitive, primitive, name='primitive')
            function = tf.multiply(switch_function, function, name='function')

        return primitive, function


class NormalModel(RandRegModel):
    def attention_generation(self, x):
        num_units = self.args.num_units
        source_sequence_length = self.x_len

        # Encoder
        if self.args.bidirectional_encoder:
            encoder_outputs, encoder_state = self.get_encoder_bidirectional(x)
        else:
            encoder_outputs, encoder_state = self.get_encoder(x)

        # Decoder
        decoder_emb_inp = self.get_decoder_input()
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        decoder_cell, encoder_state = self.get_decoder_cell(
            encoder_state, encoder_outputs, decoder_cell,
            num_units, source_sequence_length)

        ones = tf.ones(shape=self.batch_size, dtype=tf.int32)
        lengths = ones * self.max_output_length
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  encoder_state)
        outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder)
        logits = outputs.rnn_output

        score = self.ff(logits, 1, 32, self.U)
        return score

    def create_model(self):
        self.x = tf.placeholder(tf.int64, shape=(None, self.max_input_length,), name='x')
        self.y = tf.placeholder(tf.int64, shape=(None, self.max_output_length,), name='y')
        self.x_len = tf.placeholder(tf.int32, shape=(None,), name='x_len')
        self.y_len = tf.placeholder(tf.int32, shape=(None,), name='y_len')
        self.noise_weight = tf.placeholder(tf.float32, shape=(), name='noise_weight')
        self.batch_size = tf.shape(self.x_len)

        with tf.variable_scope("compute_switch"):
            switch_score = self.get_embeddings(self.x, 1)
            self.switch = tf.nn.sigmoid(
                switch_score / self.args.switch_temperature)
        # masks
        self.target_mask_float = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.float32)
        self.target_mask_int64 = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.int64)
        self.input_mask_float = tf.sequence_mask(
            self.x_len, maxlen=self.max_input_length, dtype=tf.float32)

        with tf.variable_scope("word_embeddings"):
            embedding = self.get_embeddings(self.x, 1)

        with tf.variable_scope("generate_attention"):
            self.attention = self.attention_generation(embedding)
        l = self.attention

        with tf.variable_scope("evaluation"):
            # loss
            loss_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=l)
            self.loss = tf.reduce_sum(loss_sum * self.target_mask_float / tf.to_float(self.batch_size))

            # switch regularization
            if self.args.reg_coe > 0:
                if self.args.use_entropy_reg:
                    entropy = -(self.switch * tf.log(self.switch) + (
                            1 - self.switch) * tf.log(1 - self.switch))
                else:
                    entropy = self.switch * (1 - self.switch)
                ent = tf.squeeze(entropy, axis=-1)
                reg = tf.reduce_sum(ent * self.input_mask_float / tf.to_float(self.batch_size))
                self.loss += self.args.reg_coe * reg

            if self.args.macro_switch_reg_coe > 0:
                ss = tf.squeeze(self.switch, axis=-1) - 0.5
                ss_value = tf.reduce_sum(ss * self.input_mask_float, -1) / tf.to_float(self.x_len)
                ss_reg = tf.reduce_mean(ss_value ** 2)
                self.loss += self.args.macro_switch_reg_coe * ss_reg

            for reg in self.regularization_list:
                self.loss += reg

            # word accuracy
            self.prediction = tf.argmax(l, -1) * self.target_mask_int64
            word_equality = tf.to_float(tf.equal(self.y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)
