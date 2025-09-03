import torch
def smooth_velocity_interpolation(timestamps, positions, current_time, frame_idx, method='cubic_smooth'):
    """
    多种平滑速度插值方法
    
    Args:
        timestamps: 时间戳列表
        positions: 位置列表 
        current_time: 当前时间
        frame_idx: 当前帧索引
        method: 插值方法 ('cubic_smooth', 'catmull_rom', 'moving_average')
    
    Returns:
        smooth_velocity: 平滑的速度向量
    """
    if method == 'catmull_rom':
        # Catmull-Rom样条插值
        return catmull_rom_velocity(timestamps, positions, current_time, frame_idx)
    elif method == 'moving_average':
        # 移动平均平滑
        return moving_average_velocity(timestamps, positions, current_time, frame_idx)
    else:
        # 默认使用三次平滑方法
        return cubic_smooth_velocity(timestamps, positions, current_time, frame_idx)

def catmull_rom_velocity(timestamps, positions, current_time, frame_idx):
    """Catmull-Rom样条插值速度"""
    # 获取四个控制点
    p0_idx = max(0, frame_idx - 1)
    p1_idx = frame_idx
    p2_idx = min(len(positions) - 1, frame_idx + 1)
    p3_idx = min(len(positions) - 1, frame_idx + 2)
    
    if p2_idx == p1_idx:
        # 回退到简单差分
        dt = timestamps[p2_idx] - current_time if p2_idx < len(timestamps) else 0.01
        return (positions[p2_idx] - positions[p1_idx]) / max(dt, 1e-6)
    
    # Catmull-Rom参数化
    t = (current_time - timestamps[p1_idx]) / (timestamps[p2_idx] - timestamps[p1_idx])
    t = torch.clamp(t, 0.0, 1.0)
    
    # Catmull-Rom切线计算
    tau = 0.5  # 张力参数
    
    # 计算切线
    if p0_idx < p1_idx:
        m1 = tau * (positions[p2_idx] - positions[p0_idx]) / (timestamps[p2_idx] - timestamps[p0_idx])
    else:
        m1 = (positions[p2_idx] - positions[p1_idx]) / (timestamps[p2_idx] - timestamps[p1_idx])
    
    if p3_idx > p2_idx:
        m2 = tau * (positions[p3_idx] - positions[p1_idx]) / (timestamps[p3_idx] - timestamps[p1_idx])
    else:
        m2 = (positions[p2_idx] - positions[p1_idx]) / (timestamps[p2_idx] - timestamps[p1_idx])
    
    # Hermite基函数的导数（速度）
    h00_prime = 6*t**2 - 6*t
    h10_prime = 3*t**2 - 4*t + 1
    h01_prime = -6*t**2 + 6*t
    h11_prime = 3*t**2 - 2*t
    
    dt_segment = timestamps[p2_idx] - timestamps[p1_idx]
    velocity = (h00_prime * positions[p1_idx] + h10_prime * dt_segment * m1 + 
                h01_prime * positions[p2_idx] + h11_prime * dt_segment * m2) / dt_segment
    
    return velocity

def moving_average_velocity(timestamps, positions, current_time, frame_idx, window_size=3):
    """移动平均平滑速度"""
    # 获取窗口范围
    start_idx = max(0, frame_idx - window_size//2)
    end_idx = min(len(positions) - 1, frame_idx + window_size//2)
    
    velocities = []
    weights = []
    
    for i in range(start_idx, end_idx):
        if i + 1 < len(positions) and i + 1 < len(timestamps):
            dt = timestamps[i + 1] - timestamps[i]
            if dt > 0:
                vel = (positions[i + 1] - positions[i]) / dt
                
                # 距离当前时间越近权重越大
                time_dist = abs(timestamps[i] - current_time) + 1e-6
                weight = 1.0 / (1.0 + time_dist)
                
                velocities.append(vel)
                weights.append(weight)
    
    if len(velocities) == 0:
        # 回退方案
        next_idx = min(frame_idx + 1, len(positions) - 1)
        dt = timestamps[next_idx] - current_time if next_idx < len(timestamps) else 0.01
        return (positions[next_idx] - positions[frame_idx]) / max(dt, 1e-6)
    
    # 加权平均
    total_weight = sum(weights)
    smooth_velocity = sum(w * v for w, v in zip(weights, velocities)) / total_weight
    
    return smooth_velocity

def cubic_smooth_velocity(timestamps, positions, current_time, frame_idx):

    frames_to_use = []
    times_to_use = []
    positions_to_use = []
    
    start_frame = max(0, frame_idx - 1)
    end_frame = min(len(positions) - 1, frame_idx + 2)
    
    for idx in range(start_frame, end_frame + 1):
        if idx < len(timestamps) and idx < len(positions):
            frames_to_use.append(idx)
            times_to_use.append(timestamps[idx])
            positions_to_use.append(positions[idx])
    
    if len(frames_to_use) < 2:

        next_idx = min(frame_idx + 1, len(positions) - 1)
        dt = timestamps[next_idx] - current_time if next_idx < len(timestamps) else 0.01
        return (positions[next_idx] - positions[frame_idx]) / max(dt, 1e-6)
    
    weights = []
    velocities = []
    
    for i in range(len(frames_to_use) - 1):
        curr_frame = frames_to_use[i]
        next_frame = frames_to_use[i + 1]
        
        dt_segment = timestamps[next_frame] - timestamps[curr_frame]
        if dt_segment > 0:
            velocity = (positions[next_frame] - positions[curr_frame]) / dt_segment
            
            mid_time = (timestamps[curr_frame] + timestamps[next_frame]) / 2
            time_distance = abs(mid_time - current_time) + 1e-6
            weight = 1.0 / (1.0 + time_distance)
            
            weights.append(weight)
            velocities.append(velocity)
    
    if len(velocities) == 0:
        next_idx = min(frame_idx + 1, len(positions) - 1)
        dt = timestamps[next_idx] - current_time if next_idx < len(timestamps) else 0.01
        return (positions[next_idx] - positions[frame_idx]) / max(dt, 1e-6)
    
    total_weight = sum(weights)
    smooth_velocity = sum(w * v for w, v in zip(weights, velocities)) / total_weight
    
    return smooth_velocity