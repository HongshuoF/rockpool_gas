import csv
import torch
from pathlib import Path
import config

def _load_csv_responses(csv_path, sensor_count):
    """Load the labels and sensor responses.
    Args:
        csv_path: Path to one CSV dataset in ``dat``.
        sensor_count: Number of sensors at each time point.

    Returns:
        A pair ``(labels, responses)``. ``labels`` is a list of strings and
        ``responses`` is a float tensor shaped ``[sample_count, sensor_count]``.
    """
    csv_path = Path(csv_path)

    labels = []
    selected_responses = []
    selected_start = 0
    selected_end = sensor_count

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row or all(not field.strip() for field in row):
                continue

            numeric_fields = row[1:]
            if len(numeric_fields) % sensor_count != 0:
                raise ValueError(
                    f"Row {row_number} of {csv_path.name} must contain "
                )
            time_group_count = len(numeric_fields) // sensor_count

            try:
                selected_values = [
                    float(value)
                    for value in numeric_fields[selected_start:selected_end]
                ]
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric response in {csv_path.name}, row {row_number}, "
                ) from error

            labels.append(row[0])
            selected_responses.append(selected_values)

    responses = torch.tensor(selected_responses, dtype=torch.float32)
    return labels, responses

def accumulating_rate_encoder(input, sensor_count, encoding_steps):
    """
    Encodes the input into a series of spikes using an accumulating rate coding scheme.
    Args:
        input: A float tensor shaped ``[sample_count, sensor_count]`` representing the mapped sensor responses.
        sensor_count: Number of sensors at each time point.
        encoding_steps: Number of time steps for encoding.

    Returns:
        A list of float tensors, each shaped ``[sample_count, sensor_count]``, representing the encoded spikes.
    """
    if sensor_count <= 0 or input.shape[1] % sensor_count != 0:
        raise ValueError(
            "The number of input values must be divisible by sensor_count"
        )

    outspike_list = []
    responses = input.reshape(input.shape[0], -1, sensor_count)
    input_spike = responses
    # min_input = responses.max(dim=1).values * encoding_steps
    # input_spike = 1 / (min_input + 1)
    input_accu = input_spike.clone()
    for _ in range(encoding_steps):
        # Generate rate-coded spikes using an accumulator.
        accu_temp = input_accu.clone() - 1
        accu_temp = accu_temp.floor()
        accu_temp = accu_temp.bool()
        accu_temp = ~accu_temp
        accu_temp = accu_temp.float()
        outspike_list.append(accu_temp)
        input_accu = input_accu + input_spike - accu_temp
    return outspike_list

def min_rate_encoder(input, cfg, device):
    # 找到输入中最小的值并进行累积速率编码
    outspike_list = torch.zeros((len(input), cfg.num_timesteps, cfg.sensor_num), device=device)
    batch_size = cfg.batch_size
    min_input = torch.zeros(batch_size, cfg.sensor_num, device=device)
    for i in range(len(input)):
        for j in range(cfg.sensor_num):
            min_input[i, j] = torch.max(input[i, ::cfg.sensor_num + j]) * 60 # 注：由于传感器响应最低点很低，大概在0.000x，因此为了有区分度乘了个50，有点简单粗暴了，后续可以改进    注意，依据响应曲线自行调整取最大还是最小值
    input_spike = 1 / (min_input/10+1)
   
    for j in range(len(input)): 
        input_accu = input_spike[j].clone()
        for t in range(cfg.num_timesteps):
            # changed spike encoder by input data
            accu_temp = input_accu.clone() -1
            accu_temp = accu_temp.floor()
            accu_temp = accu_temp.bool()
            accu_temp = ~accu_temp
            accu_temp = accu_temp.float()
            outspike_list[j][t] = accu_temp
            #input_accu = input_accu + input_spike * accu_temp - 1
            input_accu = input_accu + input_spike[j] - accu_temp
            # x = input_accu.clone()
            # x = x.floor()
            # x = x.bool()
            # x = ~x
            # x = x.float()
            # outspike_list.append(x)
    return outspike_list

def increase_delta_encode(input):
    """Encode responses using automatically detected response boundaries.

    ``input`` contains numeric sensor responses only and has shape
    ``[batch_size, sensor_count * T]``. Values at each time point are ordered by
    sensor. The CSV class-label column must be removed before calling this
    function. If the response pair is unsupported, ``t1 = t2 = 0`` is used;
    if the recovery pair is unsupported, ``t3 = t4 = T`` is used. 

    The code for dividing the response stage of the gas sensor is only for demonstration purposes. 
    The actual division of the sensor's response stage is carried out on the training set, and it is fixed during the actual test.
    """
    cfg = config.Config()
    sensor_count = cfg.sensor_num
    boundaries = torch.tensor(cfg.boundary_response, dtype=torch.long, device=input.device)
    if sensor_count <= 0 or input.shape[1] % sensor_count != 0:
        raise ValueError(
            "The number of input values must be divisible by sensor_count"
        )

    batch_size = input.shape[0]
    time_points = input.shape[1] // sensor_count

    responses = input.reshape(batch_size, time_points, sensor_count)

    outspike_list = []
    prediction_drift = cfg.prediction_drift
    spike_threshold = cfg.spike_threshold
    prediction = responses[:, 1, :].clone()

    for time_index in range(time_points):
        current_input = responses[:, time_index, :]
        current_delta = current_input - prediction
        current_spike = (current_delta > spike_threshold).to(input.dtype)
        current_spike -= (current_delta < -spike_threshold).to(input.dtype)
        outspike_list.append(current_spike)

        in_response_transition = (
            (time_index >= boundaries[0]) & (time_index < boundaries[1])
        )
        in_recovery_transition = (
            (time_index >= boundaries[2]) & (time_index < boundaries[3])
        )
        drift = (
            - in_response_transition.to(input.dtype)
            + in_recovery_transition.to(input.dtype)
        )* prediction_drift
        prediction = prediction + current_spike * spike_threshold + drift

    return outspike_list


def main():
    sensor_count = 6  # sensor number in dataset: 6 in example1 and 4 in example2
    time_step = 10

    csv_path = (
        Path(__file__).resolve().parents[1]
        / "dat"
        / "dat_example1(classifcation_of_wine_quality)_mapped.csv"
    )
    labels, responses = _load_csv_responses(csv_path, sensor_count)
    outspike_list = accumulating_rate_encoder(responses, sensor_count, time_step)
    return outspike_list

if __name__ == "__main__":
    main()