$$
dz^{(i)} = dy (W_3^{(i)})^T)
$$
$$
dW_3^{(i)} = (z^{(i)})^Tdy
$$
$$
dx_2^{(i)} = dz^{(i)} * f(x_1^{i})
$$
$$
dx_1^{(i)} = dz^{(i)} * f'(x_1^{i}) * x_2^{(i)}
$$
$$
dx^{(i)} = dx_1^{(i)}(W_1^{(i)})^T + dx_2^{(i)}(W_2^{(i)})^T
$$
$$
dx = \text{all-reduce}(\{dx^{(i)}\}_{i=0}^{N_{TP}-1})
$$
$$
dW_2^{(i)} = x^Tdx_2^{(i)}
$$
$$
dW_1^{(i)} = x^Tdx_1^{(i)}
$$

## 8.4(d)

Forward:
$$
N_{TP} \le \frac32 \frac{D_{FF}W}{C} + 1
$$

Backward:
$$
N_{TP} \le 3 \frac{D_{FF}W}{C} + 1
$$

## 8.5 (a)
$$
\frac{6BDD_{FF}}{N_{F}N_{T}}
$$

## 8.5 (b)

FSDP:
$$
T_{FSDP} = \frac{6(N_F-1)D_{FF}D}{WN_{F}N_T}
$$

TP:
$$
T_{TP} = \frac{4(N_T-1)BD}{WN_TN_F}
$$

$$
T_{comm} = \max(T_{FSDP}, T_{TP})
$$

## 8.5 (c)
$$
\frac{6BDD_{FF}}{N_{F}N_{T}C} \ge \frac{6(N_F-1)D_{FF}D}{WN_{F}N_T} \Rightarrow N_F \le \frac{BW}{C} + 1
$$

$$
\frac{6BDD_{FF}}{N_{F}N_{T}C} \ge \frac{4(N_T-1)BD}{WN_TN_F} \Rightarrow N_T \le \frac{3D_{FF}W}{2C} + 1
$$

$$
N \le (\frac{BW}{C} + 1)(\frac{3D_{FF}W}{2C} + 1)
$$

# 8.5 (d)

$$
\frac{6BDD_{FF}}{N_{F}N_{T}C} \ge \frac{6(N_F-1)D_{FF}D}{WN_{F}N_T} + \frac{4(N_T-1)BD}{WN_TN_F} 
$$
$$
3N_FD_{FF} + 2N_TB \le \frac{3BD_{FF}W}{C} + 3D_{FF} + 2B
$$
The maximum of $N_FN_T$ is obtained when 
$$
3N_FD_{FF} = 2N_TB = \frac{3BD_{FF}W}{2C} + \frac{3D_{FF}}{2} +B
$$
$$
\Rightarrow N_F = \frac{BW}{2C} + \frac{1}{2} + \frac{B}{3D_{FF}},\quad N_T = \frac{3D_{FF}W}{4C} + \frac{3D_{FF}}{4B} + \frac{1}{2}
$$
$$
N = N_FN_T
$$



