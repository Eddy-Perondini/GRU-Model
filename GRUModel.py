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

    @tf.function
    def forward(self, inputs, H=None):
        """
        inputs: Tensor shape (seq_len, batch_size, num_inputs)
        H: estado escondido inicial (batch_size, num_hiddens)
        """

        if H is None:
            H = tf.zeros((inputs.shape[1], self.num_hiddens))

        outputs = []

        for X in inputs:  # tamanho de X: (batch_size, num_inputs)
            Z = tf.sigmoid(tf.matmul(X, self.W_xz) +
                           tf.matmul(H, self.W_hz) + self.b_z)

            R = tf.sigmoid(tf.matmul(X, self.W_xr) +
                           tf.matmul(H, self.W_hr) + self.b_r)

            H_tilde = tf.tanh(tf.matmul(X, self.W_xh) +
                              tf.matmul(R * H, self.W_hh) + self.b_h)

            H = Z * H + (1 - Z) * H_tilde
            outputs.append(H)

        return outputs, H
    
class GRUModel(): 
    def __init__(self, num_inputs, num_hiddens, num_outputs):
        self.gru = GRUScratch(num_inputs, num_hiddens)
        self.W_hq = tf.Variable(tf.random.normal((num_hiddens, num_outputs)) * 0.01)
        self.b_q = tf.Variable(tf.zeros(num_outputs))

    def __call__(self, X): 
        outputs, H = self.gru.forward(X)
        return tf.matmul(H, self.W_hq) + self.b_q 
    
        
#Função de perda + Otimizador

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-3)

#Treinamento do modelo da GRU

@tf.function
def train_step(model, X, y): 
    with tf.GradientTape() as tape: 
        y_pred = model(X)
        loss = loss_fn(y, y_pred)

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss 