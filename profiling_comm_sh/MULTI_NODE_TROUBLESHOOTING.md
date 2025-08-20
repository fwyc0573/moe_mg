# 双机AllReduce测试故障排除指南

## 🚨 问题症状分析

### 您遇到的问题特征：
- ✅ Master节点正常启动并开始测试
- ❌ Worker节点启动延迟4分钟
- ❌ Worker节点无测试输出，只有OMP警告
- ❌ Master节点在32K数据后停止进展
- ❌ 进程挂起，无进一步日志

## 🔍 根因分析

### 1. 网络连接问题
- **端口阻塞**: 29500端口可能被防火墙阻塞
- **网络延迟**: InfiniBand配置问题导致连接超时
- **IP路由**: 节点间无法正确路由

### 2. NCCL初始化超时
- **Rendezvous超时**: PyTorch分布式初始化失败
- **CUDA上下文**: GPU设备初始化问题
- **同步阻塞**: `dist.barrier()`等待所有节点就绪

## 🛠️ 解决方案步骤

### 步骤1：网络连接验证

```bash
# 在两个节点上运行网络诊断脚本
chmod +x debug_network_connectivity.sh
./debug_network_connectivity.sh
```

**关键检查项：**
- [ ] 基础网络连通性（ping测试）
- [ ] 端口可用性（29500-29510）
- [ ] 防火墙配置
- [ ] InfiniBand状态

### 步骤2：使用简化连接测试

```bash
# Master节点 (10.232.194.226)
python simple_connection_test.py --rank 0 --world_size 2 --master_addr 10.232.194.226 --master_port 29501

# Worker节点 (10.232.195.25) - 在另一个终端同时运行
python simple_connection_test.py --rank 1 --world_size 2 --master_addr 10.232.194.226 --master_port 29501
```

### 步骤3：使用改进的测试脚本

```bash
# Master节点
chmod +x robust_multi_node_allreduce.sh
./robust_multi_node_allreduce.sh --master --gpus 2 --port 29501

# Worker节点
./robust_multi_node_allreduce.sh --worker --gpus 2 --port 29501
```

## 🔧 环境变量优化

### 必需的NCCL环境变量

```bash
# 基础配置
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 超时配置
export NCCL_SOCKET_TIMEOUT=60000
export NCCL_CONNECT_TIMEOUT=60000
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# InfiniBand配置
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5

# 网络接口配置（根据实际情况调整）
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=3
```

### PyTorch分布式配置

```bash
# 增加超时时间
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_BLOCKING_WAIT=1

# 减少内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 🌐 网络配置检查

### 防火墙配置

```bash
# Master节点开放端口
sudo ufw allow from 10.232.195.25 to any port 29500:29510
sudo ufw reload

# Worker节点允许出站连接
sudo ufw allow out to 10.232.194.226 port 29500:29510
```

### InfiniBand配置验证

```bash
# 检查IB设备状态
ibstat
ibportstate

# 测试IB连接
# Master节点
ibping -S

# Worker节点
ibping -c 10.232.194.226
```

## 🐛 调试技巧

### 1. 分阶段测试

```bash
# 阶段1: 基础连接测试
python simple_connection_test.py

# 阶段2: 小规模AllReduce测试
# 使用 --max_size 1M --iterations 3

# 阶段3: 完整测试
# 逐步增加数据大小和迭代次数
```

### 2. 日志分析

```bash
# 启用详细日志
export NCCL_DEBUG=TRACE
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 分别保存Master和Worker日志
torchrun ... 2>&1 | tee master_debug.log
torchrun ... 2>&1 | tee worker_debug.log
```

### 3. 进程监控

```bash
# 监控网络连接
watch -n 1 'netstat -tuln | grep 29500'

# 监控GPU使用
watch -n 1 nvidia-smi

# 监控进程状态
watch -n 1 'ps aux | grep torchrun'
```

## 🚀 快速修复方案

### 方案A：使用不同端口

```bash
# 如果29500被占用，使用29501
--master_port=29501
```

### 方案B：减少测试规模

```bash
# 减少数据大小和迭代次数进行初步测试
--max_size=1M --iterations=3 --warmup_iters=1
```

### 方案C：强制使用以太网

```bash
# 如果InfiniBand有问题，临时使用以太网
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

## 📋 检查清单

在运行测试前，确保完成以下检查：

- [ ] 两个节点都能ping通对方
- [ ] 端口29500-29510在两个节点都可用
- [ ] 防火墙允许相应端口通信
- [ ] InfiniBand设备状态正常
- [ ] CUDA和PyTorch环境一致
- [ ] 时间同步（NTP）
- [ ] 相同的Python脚本版本
- [ ] 足够的GPU内存

## 🆘 紧急恢复

如果所有方案都失败，尝试以下紧急恢复步骤：

1. **重启网络服务**：
   ```bash
   sudo systemctl restart networking
   sudo systemctl restart openibd  # InfiniBand
   ```

2. **清理PyTorch缓存**：
   ```bash
   rm -rf ~/.cache/torch_extensions/
   ```

3. **使用TCP后端**：
   ```bash
   # 在Python脚本中临时使用TCP而不是NCCL
   dist.init_process_group(backend='gloo')
   ```

4. **单机测试验证**：
   ```bash
   # 先在单机上验证脚本正常工作
   ./single_node_allreduce.sh --2gpu
   ```

## 📞 获取帮助

如果问题仍然存在，请收集以下信息：

1. 网络诊断脚本输出
2. 简化连接测试结果
3. 完整的错误日志（Master和Worker）
4. 系统配置信息（网络、GPU、NCCL版本）
5. InfiniBand状态信息
