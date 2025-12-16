Modeling a GRU for sequential data prediction using Tensorflow.

BRIEF EXPLANATION:

\textbf{Building Gated Recurrent Units (GRUs) — an improved version of RNNs and LSTMs}

\medskip

\textbf{MAIN PROBLEM OF RNNs:} Vanishing Gradient

\medskip

\textbf{ADVANTAGES OF GRUs:}
\begin{itemize}
    \item Fewer parameters to compute;
    \item Lower computational complexity compared to RNNs and LSTMs;
    \item Lower memory usage.
\end{itemize}

\medskip
\hrule
\medskip

\textbf{GRU ARCHITECTURE}

Let $f$ be an activation function, usually represented by
$f = \tanh(\cdot)$ or $f = \sigma(\cdot)$.

\medskip

\textbf{Update Gate} — Decides which information to keep or discard /  
Determines how much information is retained in the processing cell:
\[
Z_t = f\left(W_{xz} X_t + W_{hz} h_{t-1} + b\right)
\]

\medskip

\textbf{Reset Gate} — Decides how much information is retained within a processing cell /  
Adjusts inputs based on previous memories:
\[
r_t = f\left(W_{xr} X_t + W_{hr} h_{t-1} + b\right)
\]

\medskip

\textbf{Candidate Hidden State} — Selection of candidate memories for updating the cells:
\[
\hat{h}_t = f\left(W_{xh} X_t + W_{hh} \left(r_t \odot h_{t-1}\right) + b\right)
\]

\medskip

\textbf{Hidden State} — Capable of mitigating the vanishing gradient problem in RNNs:
\[
h_t = (1 - Z_t) \odot h_{t-1} + Z_t \odot \hat{h}_t
\]

\medskip
\hrule
\medskip

\textbf{References for Model Construction:}
\begin{itemize}
    \item \url{https://github.com/d2l-ai/d2l-tensorflow-colab/blob/master/chapter_recurrent-modern/gru.ipynb}
    \item \url{https://www.youtube.com/watch?v=rdz0UqQz5Sw}
\end{itemize}

