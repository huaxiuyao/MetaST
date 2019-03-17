import tensorflow as tf
import math

class T_LSTM_AE(object):
    def init_weights(self, input_dim, output_dim, name=None, std=1.0):
        return tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=std / math.sqrt(input_dim)), name=name)

    def init_bias(self, output_dim, name=None):
        return tf.Variable(tf.zeros([output_dim]), name=name)

    def __init__(self, inputs, input_dim, output_dim, output_dim2, output_dim3, hidden_dim, hidden_dim_dec, hidden_dim3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim2 = output_dim2
        self.output_dim3 = output_dim3
        self.hidden_dim = hidden_dim
        self.hidden_dim_dec = hidden_dim_dec
        self.hidden_dim3 = hidden_dim3

        self.Wi_enc = self.init_weights(input_dim, hidden_dim, name='Input_Hidden_weight_enc')
        self.Ui_enc = self.init_weights(hidden_dim, hidden_dim, name='Input_State_weight_enc')
        self.bi_enc = self.init_bias(hidden_dim, name='Input_Hidden_bias_enc')

        self.Wf_enc = self.init_weights(input_dim, hidden_dim, name='Forget_Hidden_weight_enc')
        self.Uf_enc = self.init_weights(hidden_dim, hidden_dim, name='Forget_State_weight_enc')
        self.bf_enc = self.init_bias(hidden_dim, name='Forget_Hidden_bias_enc')

        self.Wog_enc = self.init_weights(input_dim, hidden_dim, name='Output_Hidden_weight_enc')
        self.Uog_enc = self.init_weights(hidden_dim, hidden_dim, name='Output_State_weight_enc')
        self.bog_enc = self.init_bias(hidden_dim, name='Output_Hidden_bias_enc')

        self.Wc_enc = self.init_weights(input_dim, hidden_dim, name='Cell_Hidden_weight_enc')
        self.Uc_enc = self.init_weights(hidden_dim, hidden_dim, name='Cell_State_weight_enc')
        self.bc_enc = self.init_bias(hidden_dim, name='Cell_Hidden_bias_enc')

        self.Wi_dec = self.init_weights(input_dim, hidden_dim_dec, name='Input_Hidden_weight_dec')
        self.Ui_dec = self.init_weights(hidden_dim_dec, hidden_dim_dec, name='Input_State_weight_dec')
        self.bi_dec = self.init_bias(hidden_dim_dec, name='Input_Hidden_bias_dec')

        self.Wf_dec = self.init_weights(input_dim, hidden_dim_dec, name='Forget_Hidden_weight_dec')
        self.Uf_dec = self.init_weights(hidden_dim_dec, hidden_dim_dec, name='Forget_State_weight_dec')
        self.bf_dec = self.init_bias(hidden_dim_dec, name='Forget_Hidden_bias_dec')

        self.Wog_dec = self.init_weights(input_dim, hidden_dim_dec, name='Output_Hidden_weight_dec')
        self.Uog_dec = self.init_weights(hidden_dim_dec, hidden_dim_dec, name='Output_State_weight_dec')
        self.bog_dec = self.init_bias(hidden_dim_dec, name='Output_Hidden_bias_dec')

        self.Wc_dec = self.init_weights(input_dim, hidden_dim_dec, name='Cell_Hidden_weight_dec')
        self.Uc_dec = self.init_weights(hidden_dim_dec, hidden_dim_dec, name='Cell_State_weight_dec')
        self.bc_dec = self.init_bias(hidden_dim_dec, name='Cell_Hidden_bias_dec')

        self.Wo3 = self.init_weights(hidden_dim3, output_dim3, name='Output_Layer_weight_dec2')
        self.bo3 = self.init_bias(output_dim3, name='Output_Layer_bias_dec2')

        # [batch size x seq length x input dim]
        self.input = inputs


    def T_LSTM_Encoder_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0,0], [batch_size, self.input_dim])
        i = tf.sigmoid(tf.matmul(x, self.Wi_enc) + tf.matmul(prev_hidden_state, self.Ui_enc) + self.bi_enc)
        f = tf.sigmoid(tf.matmul(x, self.Wf_enc) + tf.matmul(prev_hidden_state, self.Uf_enc) + self.bf_enc)
        o = tf.sigmoid(tf.matmul(x, self.Wog_enc) + tf.matmul(prev_hidden_state, self.Uog_enc) + self.bog_enc)

        C = tf.nn.tanh(tf.matmul(x, self.Wc_enc) + tf.matmul(prev_hidden_state, self.Uc_enc) + self.bc_enc)

        Ct = f * prev_cell + i * C

        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])


    def T_LSTM_Decoder_Unit(self, prev_hidden_memory, concat_input):
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 0], [batch_size, self.input_dim])
        i = tf.sigmoid(tf.matmul(x, self.Wi_dec) + tf.matmul(prev_hidden_state, self.Ui_dec) + self.bi_dec)
        f = tf.sigmoid(tf.matmul(x, self.Wf_dec) + tf.matmul(prev_hidden_state, self.Uf_dec) + self.bf_dec)
        o = tf.sigmoid(tf.matmul(x, self.Wog_dec) + tf.matmul(prev_hidden_state, self.Uog_dec) + self.bog_dec)

        C = tf.nn.tanh(tf.matmul(x, self.Wc_dec) + tf.matmul(prev_hidden_state, self.Uc_dec) + self.bc_dec)

        Ct = f * prev_cell + i * C

        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_encoder_states(self): # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_) #scan input is [seq_length x batch_size x input_dim]
        # scan_time = tf.transpose(self.time) # scan_time [seq_length x batch_size]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])
        packed_hidden_states = tf.scan(self.T_LSTM_Encoder_Unit, scan_input, initializer=ini_state_cell, name='encoder_states')
        all_encoder_states = packed_hidden_states[:, 0, :, :]
        all_encoder_cells = packed_hidden_states[:, 1, :, :]
        return all_encoder_states, all_encoder_cells

    def get_representation(self):
        all_encoder_states, all_encoder_cells = self.get_encoder_states()
        # We need the last hidden state of the encoder
        representation = tf.reverse(all_encoder_states, [0])[0, :,:]
        decoder_ini_cell = tf.reverse(all_encoder_cells, [0])[0, :, :]
        return representation, decoder_ini_cell

    # def get_output(self, state):
    #     output = tf.matmul(state, self.Wo) + self.bo
    #     return output
    # def get_output2(self, state):
    #     output = tf.matmul(state, self.Wo2) + self.bo2
    #     return output
    def get_output3(self, state):
        output = tf.matmul(state, self.Wo3) + self.bo3
        return output

    def get_decoder_states(self):
        #bug
        batch_size = tf.shape(self.input)[0]
        seq_length = tf.shape(self.input)[1]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        scan_input = tf.slice(scan_input, [0,0,0],[seq_length ,batch_size, self.input_dim])
        scan_input = tf.reverse(scan_input, [0])
        initial_hidden, initial_cell = self.get_representation()
        ini_state_cell = tf.stack([initial_hidden, initial_cell])
        packed_hidden_states = tf.scan(self.T_LSTM_Decoder_Unit, scan_input, initializer=ini_state_cell, name='decoder_states')
        all_decoder_states = packed_hidden_states[:, 0, :, :]
        return all_decoder_states

    def get_decoder_outputs(self): # Returns the output of only the last time step
        all_decoder_states = self.get_decoder_states()
        all_outputs = tf.map_fn(self.get_output3, all_decoder_states)
        reversed_outputs = tf.reverse(all_outputs, [0])
        outputs_ = tf.transpose(reversed_outputs, perm=[2, 0, 1])
        outputs = tf.transpose(outputs_)
        return outputs

    def get_reconstruction_loss(self):
        outputs = self.get_decoder_outputs()
        loss = tf.reduce_mean(tf.square(self.input - outputs))
        return loss