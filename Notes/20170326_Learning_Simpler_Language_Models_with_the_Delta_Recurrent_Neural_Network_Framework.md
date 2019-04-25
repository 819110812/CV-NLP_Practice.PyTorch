# [Learning Simpler Language Models with the Delta Recurrent Neural Network Framework](https://128.84.21.199/pdf/1703.08864.pdf)

The Delta-RNN models maintain longer-term memory by learning to interpolate between a fast-changing data-driven representation and a slowly changing, implicitly stable state.

```mermaid
graph LR;
    style H_t_perv fill:#f9f,stroke:#333,stroke-width:4px;
    style IN_word fill:#f9f,stroke:#333,stroke-width:4px;
    style H_t_out fill:#f9f,stroke:#333,stroke-width:4px;
    style Next_wp fill:#f9f,stroke:#333,stroke-width:4px;

    H_t_perv[hidden t-1]==>Gate_a[Gate gamma];
    IN_word[Current input word]==>G_theta
    H_t_perv==>G_theta
    G_theta==>Gate_b[Gate beta];G_theta-.->Gate_b;G_theta-.->Gate_a
    Gate_a==>H_t_out[Hidden t];
    Gate_b==>H_t_out;H_t_out==>Next_wp[Next word]
```

- $g_{\theta}$ function maps the previous hidden state and the currently encountered data point (e.g.  a word, subword, or character token) to a real-valued vector of fixed dimensions using parameters;
