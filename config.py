# config.py
from ast import Constant
from dataclasses import dataclass, field

@dataclass
class Config:
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 5000    
    dataset_root: str = "D:\\gas_dataset\\"
    ## NOTE:
    # dataset define with data_input.py
    # 默认数据集形式如下： 标签 时间1的全部传感数据（传感器1 传感器2 ... 传感器n） 时间2 ... 时间m
    # 该程序主要功能依据传感器数量对数据进行划分，按照时间顺序将其输入主程序
    # 以下是不同传感器数据集对应路径
    # dataset_path: str = "twin+gas+sensor+arrays/data_processedV3/20251224.csv"
    # labellist: dict = field(default_factory=lambda: {'CO': 0, 'Ey': 1, 'Ea': 2, 'Me': 3})
    # sensor_num: int = 8       # 每条数据有几个传感器数据
    # sensor_start: int = 1     # 从第几个数据开始是传感器数据

    dataset_path: str = "wine/data_processV2/2025112201.csv"
    sensor_num: int = 6         # 每条数据有几个传感器数据
    sensor_start: int = 1     # 从第几个-1开始是传感器数据
    labellist: dict = field(default_factory=lambda: {'HQ': 0, 'AQ': 1, 'LQ': 2, 'Ea': 3})

    # dataset_path: str = "WTD/2025112401.csv"
    # sensor_num: int = 72       # 每条数据有几个传感器数据
    # sensor_start: int = 1     # 从第几个-1开始是传感器数据
    # labellist: dict = field(default_factory=lambda: {'Acetaldehyde': 0, 'Acetone': 1, 'Ammonia': 2, 'Benzene': 3, 'Butanol': 4, 'CO': 5, 'Ethylene': 6, 'Methane': 7, 'Methanol': 8, 'Toluene': 9})

    # dataset_path: str = "self_construction/data_processedV1/data2.csv"
    # sensor_num: int = 4        # 每条数据有几个传感器数据
    # sensor_start: int = 1     # 从第几个-1开始是传感器数据
    # labellist: dict = field(default_factory=lambda: {'5': 0, '30': 1, '60': 2})
    category_num: int = field(init=False)
    hidden_dim: list = field(init=False)
    seed: int = 42
    num_timesteps: int = 30
    time_all: int = 30    
    
    # - Define the temporal aspects of a data sample
    dt: float = 10e-3

    # The following parameters are used to define the LIF neuron model b1 means branch 1(connected to rate encoder)
    # b2 means branch 2(connected to delta encoder)
    tau_mem: list = field(init=False)  # Membrane time constant for branch 1,2,shared
    tau_syn: list = field(init=False)  # Synaptic time constant for branch 1,2,shared
    threshold: list = field(init=False)  # Firing threshold for branch 1,2,shared
    bias: list = field(init=False)  # Bias for branch 1,2,shared

    # - Define the inhibitory connection probability and strength
    p_inhib: float = 0.3
    w_inhib: float = -1.0
    inhibit_num_per_neuron: int = 5

    # gas data divide point t1,t2,t3,t4, t1-t2: response stage, t3-t4: recovery stage, t2-t3: steady stage
    boundary_response: list = field(init=False)
    prediction_drift = 0.05
    spike_threshold = 0.03

    def __post_init__(self):
        self.category_num = len(self.labellist)
        self.hidden_dim = [self.sensor_num, 32, self.category_num]
        # self.boundary_response = [0, 0, 30, 30] # WTD
        self.boundary_response = [0, 6, 30, 30] # wine
        # self.boundary_response = [7, 15, 26, 48] # twin
        # self.boundary_response = [1, 17, 30, 30] # self
        self.tau_mem = [100e-3, 100e-3, 100e-3]  # Membrane time constant for branch 1,2,shared
        self.tau_syn = [50e-3, 50e-3, 50e-3]  # Synaptic time constant for branch 1,2,shared
        self.threshold = [1., 1., 1.]  # Firing threshold for branch 1,2,shared
        self.bias = [0., 0., 0.]  # Bias for branch 1,2,shared
