'''
Construindo a Gated Recurrent Units (GRUs) - uma versão melhorada das RNNs e LSTMs

PROBLEMA PRINCIPAL DAS RNNs: Gradient Vanishing 

VANTAGEM DA GRU:

    - Possui menos parâmetros para computar;
    - Menor complexidade computacional em relação as RNNs e as LSTMs;
    - Utiliza menos memória. 

----- ARQUITETURA DAS GRUs -----

Seja f uma função de ativação, onde usualmente é representado pela função de 
ativação f  = tanh() ou f = sigmoid().

Portão de Atualização - Decide que tipo de informação manter ou descartar / 
                        Determina quanta informação é mantida na célula de processamento

    Z_{t} = f(W_{xz}X_{t} + W_{hz}h_{t-1} + b)

Portão de Reinicialização - Decide quanta informação é mantida dentro  de uma célula de processamento /
                            Ajusta os inputs através de memórias anteriores

    r_{t} = f(W_{xr}X_{t} + W_{hr}h_{t-1} + b)

Estado Escondido Candidato - Seleção de memórias candidatas a atualização das células

    \hat{h}_{t} = f(W_{xh}X_{t} + W_{hh} (r_{t} * h_{t-1}) + b) 

Estado Escondido - Capaz de resolver o problema do gradient vanishing das RNNs

    h_{t} = (1 - Z_{t}) * h_{t-1} + Z_{t} * \hat{h}_{t}

--------------------------------

Referências p/ Construção do Modelo: 
https://github.com/d2l-ai/d2l-tensorflow-colab/blob/master/chapter_recurrent-modern/gru.ipynb
https://www.youtube.com/watch?v=rdz0UqQz5Sw

'''

import tensorflow as tf

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

class GRUScratch:
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        
        init_weight = lambda *shape: tf.Variable(tf.random.normal(shape) * sigma)
        triple = lambda: (
            init_weight(num_inputs, num_hiddens),
            init_weight(num_hiddens, num_hiddens),
            tf.Variable(tf.zeros(num_hiddens))
        )

        # Portão de Atualização
        self.W_xz, self.W_hz, self.b_z = triple()
        # Portão de Reinicialização
        self.W_xr, self.W_hr, self.b_r = triple()
        # Estado Escondido Candidato
        self.W_xh, self.W_hh, self.b_h = triple()

   
    def forward(self, inputs, H=None):
        """
        inputs: Tensor shape (seq_len, batch_size, num_inputs)
        H: estado escondido inicial (batch_size, num_hiddens)
        """

        if H is None:
            H = tf.zeros((inputs.shape[1], self.num_hiddens))

        outputs = []

        for X in tf.unstack(inputs, axis=0):  # tamanho de X: (batch_size, num_inputs)
            Z = tf.sigmoid(tf.matmul(X, self.W_xz) +
                           tf.matmul(H, self.W_hz) + self.b_z)

            R = tf.sigmoid(tf.matmul(X, self.W_xr) +
                           tf.matmul(H, self.W_hr) + self.b_r)

            H_tilde = tf.tanh(tf.matmul(X, self.W_xh) +
                              tf.matmul(R * H, self.W_hh) + self.b_h)

            H = (1 - Z) * H + Z * H_tilde
            outputs.append(H)
            
        return tf.stack(outputs)
    
class GRUModel(): 
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.gru = GRUScratch(num_inputs, num_hiddens)
        self.W_hq = tf.Variable(tf.random.normal((num_hiddens, num_outputs)) * 0.01)
        self.b_q = tf.Variable(tf.zeros(num_outputs))

    def __call__(self, X): 
        Hs = self.gru.forward(X)
        Y = tf.matmul(Hs, self.W_hq) + self.b_q
        return Y
    
        
    @property
    def trainable_variables(self):
        return (
            self.gru.W_xz, self.gru.W_hz, self.gru.b_z,
            self.gru.W_xr, self.gru.W_hr, self.gru.b_r,
            self.gru.W_xh, self.gru.W_hh, self.gru.b_h,
            self.W_hq, self.b_q
        )

@tf.function
def train_step(model, X, Y):

    with tf.GradientTape() as tape:
        Y_hat = model(X)
        loss = loss_fn(Y, Y_hat)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss