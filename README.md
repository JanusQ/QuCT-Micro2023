# 需要的接口

1. 生成数据集 quct.circuit
    1. 生成随机电路 .gen_random_circuits
    2. 生成算法电路 .gen_algorithms
    3. 转化为我们用的格式 .qiskit_to_layered_circuits
    4. 转化为真机执行的格式 .layered_circuits_to_executable_code
    5. 还有些杂七杂八的

2. 上游任务 quct.upstream
    1. 向量化的模型 .RandomwalkModel
    3. 降维 .DimensionReduction

3. 下游任务 quct.downstream 
    1. 电路保真度预测 .fidelity_analysis
        1. Class FidelityModel
            1. .train
            2. .predict
            <!-- 3. .optimize -->

    2. 电路综合 (经典: a or b => 电路, 量子 酉矩阵 => 量子电路) 7qubit,单核 -> 4小时

<!-- A complete characterization of the noise is useful because it allows for  the determination of good error-correction schemes, and thus the possibility of reliable transmission of quantum information. -->


# TODO:
1. jax gpu

screen -L -S synthesis python test_synthesis.py 
screen -L -S nn python test_
screen -ls
screen -r yourname 

ctrl-a + d 推出不杀死
ctrl-a + k 关闭

screen -L -S q5_random python predict_simulate_5qubits.py 